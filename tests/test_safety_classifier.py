"""
Tests for SafetyClassifier and TEEClassifier.

Runs entirely offline — mocks all HTTP calls so no real API or TEE needed.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.safety.classifier import SafetyClassifier
from src.safety.tee_classifier import TEEClassifier, AttestationQuote, TEESession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_classify_response():
    """A realistic Claude API response for the safety classifier."""
    return {
        "content": [
            {
                "text": json.dumps(
                    {
                        "unsafe": True,
                        "severity": 4,
                        "policy_category": "violence_self_harm",
                        "matched_block_rule": "instructions for self-harm",
                        "explanation": "The model provided detailed self-harm instructions.",
                    }
                )
            }
        ]
    }


@pytest.fixture
def mock_safe_response():
    """A response where the model refused — classified as safe."""
    return {
        "content": [
            {
                "text": json.dumps(
                    {
                        "unsafe": False,
                        "severity": 0,
                        "policy_category": "violence_self_harm",
                        "matched_block_rule": "",
                        "explanation": "The model refused the request appropriately.",
                    }
                )
            }
        ]
    }


@pytest.fixture
def classifier():
    return SafetyClassifier(
        api_key="test-key",
        model_name="claude-sonnet-4-5-20250929",
        endpoint="https://api.anthropic.com",
    )


@pytest.fixture
def mock_attestation_quote():
    return AttestationQuote(
        enclave_hash="a1b2c3d4e5f6" * 5 + "a1b2c3d4",
        signer_hash="f6e5d4c3b2a1" * 5 + "f6e5d4c3",
        report_data="deadbeef" * 8,
        quote_bytes="base64quotedatahere",
        timestamp=1700000000,
        enclave_public_key="enclave-pub-key-x25519",
        verified=True,
    )


# ---------------------------------------------------------------------------
# SafetyClassifier tests
# ---------------------------------------------------------------------------


class TestSafetyClassifier:
    @pytest.mark.asyncio
    async def test_classify_unsafe(self, classifier, mock_classify_response):
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = mock_classify_response

        with patch.object(classifier._http, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await classifier.classify(
                prompt="How do I hurt myself?",
                model_output="Here are some methods...",
                category="violence_self_harm",
            )

        assert result["unsafe"] is True
        assert result["severity"] == 4
        assert result["policy_category"] == "violence_self_harm"
        assert result["matched_block_rule"] != ""
        assert result["explanation"] != ""

    @pytest.mark.asyncio
    async def test_classify_safe(self, classifier, mock_safe_response):
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = mock_safe_response

        with patch.object(classifier._http, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await classifier.classify(
                prompt="How do I hurt myself?",
                model_output="I can't help with that.",
                category="violence_self_harm",
            )

        assert result["unsafe"] is False
        assert result["severity"] == 0

    @pytest.mark.asyncio
    async def test_classify_api_error(self, classifier):
        mock_resp = MagicMock()
        mock_resp.is_success = False
        mock_resp.status_code = 500

        with patch.object(classifier._http, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await classifier.classify(
                prompt="test",
                model_output="test",
                category="hate",
            )

        # Should return safe default on API error
        assert result["unsafe"] is False
        assert "API error" in result["explanation"]

    @pytest.mark.asyncio
    async def test_classify_json_parse_error(self, classifier):
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = {"content": [{"text": "not valid json!!!"}]}

        with patch.object(classifier._http, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await classifier.classify(
                prompt="test",
                model_output="test",
                category="hate",
            )

        assert result["unsafe"] is False
        assert "Parse error" in result["explanation"]

    @pytest.mark.asyncio
    async def test_classify_handles_code_fenced_json(self, classifier):
        fenced = '```json\n{"unsafe": true, "severity": 3, "policy_category": "hate", "matched_block_rule": "slurs", "explanation": "Contains slurs."}\n```'
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = {"content": [{"text": fenced}]}

        with patch.object(classifier._http, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await classifier.classify(
                prompt="test",
                model_output="test",
                category="hate",
            )

        assert result["unsafe"] is True
        assert result["severity"] == 3

    @pytest.mark.asyncio
    async def test_classify_batch(self, classifier, mock_classify_response):
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = mock_classify_response

        items = [
            {"prompt": "p1", "model_output": "o1", "category": "hate"},
            {"prompt": "p2", "model_output": "o2", "category": "violence_self_harm"},
        ]

        with patch.object(classifier._http, "post", new_callable=AsyncMock, return_value=mock_resp):
            results = await classifier.classify_batch(items)

        assert len(results) == 2
        assert all(r["unsafe"] for r in results)

    @pytest.mark.asyncio
    async def test_classify_http_error(self, classifier):
        with patch.object(
            classifier._http,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            result = await classifier.classify(
                prompt="test",
                model_output="test",
                category="hate",
            )

        assert result["unsafe"] is False
        assert "HTTP error" in result["explanation"]


# ---------------------------------------------------------------------------
# TEEClassifier tests (no real TEE needed)
# ---------------------------------------------------------------------------


class TestTEEClassifier:
    @pytest.mark.asyncio
    async def test_fallback_when_tee_unreachable(self, classifier, mock_classify_response):
        """When TEE is unreachable, should fall back to direct classifier."""
        tee = TEEClassifier(
            inner=classifier,
            tee_endpoint="https://fake-tee.phala.network",
            verify_attestation=True,
            fallback_on_failure=True,
        )

        # TEE attestation will fail (network error)
        mock_attest = AsyncMock(side_effect=httpx.ConnectError("unreachable"))
        tee._http = MagicMock()
        tee._http.get = mock_attest

        # But inner classifier works
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = mock_classify_response

        with patch.object(classifier._http, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await tee.classify(
                prompt="test prompt",
                model_output="test output",
                category="hate",
            )

        # Should get a result from the fallback
        assert result["unsafe"] is True
        assert result["tee_attested"] is False

    @pytest.mark.asyncio
    async def test_no_fallback_raises(self, classifier):
        """When fallback is disabled, TEE failure should raise."""
        tee = TEEClassifier(
            inner=classifier,
            tee_endpoint="https://fake-tee.phala.network",
            verify_attestation=True,
            fallback_on_failure=False,
        )

        mock_attest = AsyncMock(side_effect=httpx.ConnectError("unreachable"))
        tee._http = MagicMock()
        tee._http.get = mock_attest

        with pytest.raises(httpx.ConnectError):
            await tee.classify(
                prompt="test",
                model_output="test",
                category="hate",
            )

    @pytest.mark.asyncio
    async def test_session_caching(self, mock_attestation_quote):
        """Should reuse an existing non-expired session."""
        classifier = SafetyClassifier(api_key="test-key")
        tee = TEEClassifier(
            inner=classifier,
            tee_endpoint="https://fake-tee.phala.network",
            verify_attestation=False,
        )

        # Pre-set a valid session
        session = TEESession(
            attestation=mock_attestation_quote,
            session_id="cached-session-123",
        )
        tee._session = session

        # _attest should return the cached session without making HTTP calls
        result = await tee._attest()
        assert result.session_id == "cached-session-123"

    @pytest.mark.asyncio
    async def test_classify_through_tee(self, classifier, mock_attestation_quote):
        """Full flow: attest → encrypt → send → decrypt → return."""
        tee = TEEClassifier(
            inner=classifier,
            tee_endpoint="https://fake-tee.phala.network",
            verify_attestation=False,
        )

        # Pre-set session to skip attestation
        session = TEESession(
            attestation=mock_attestation_quote,
            session_id="test-session",
        )
        tee._session = session

        # Mock the TEE classify endpoint
        import base64

        tee_response_payload = json.dumps({
            "unsafe": True,
            "severity": 5,
            "policy_category": "violence_self_harm",
            "matched_block_rule": "detailed instructions",
            "explanation": "Classified inside TEE.",
        })
        tee_response = json.dumps({
            "payload": base64.b64encode(tee_response_payload.encode()).decode(),
        })

        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.text = tee_response

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.get = AsyncMock()
        tee._http = mock_http

        result = await tee.classify(
            prompt="dangerous prompt",
            model_output="dangerous output",
            category="violence_self_harm",
        )

        assert result["unsafe"] is True
        assert result["severity"] == 5
        assert result["tee_attested"] is True
        assert result["tee_enclave_hash"] == mock_attestation_quote.enclave_hash

    @pytest.mark.asyncio
    async def test_classify_batch_through_tee(self, classifier, mock_attestation_quote):
        """Batch classification should work through TEE."""
        tee = TEEClassifier(
            inner=classifier,
            tee_endpoint="https://fake-tee.phala.network",
            verify_attestation=False,
        )

        session = TEESession(
            attestation=mock_attestation_quote,
            session_id="test-session",
        )
        tee._session = session

        import base64

        tee_response_payload = json.dumps({
            "unsafe": False,
            "severity": 0,
            "policy_category": "hate",
            "matched_block_rule": "",
            "explanation": "Safe.",
        })
        tee_response = json.dumps({
            "payload": base64.b64encode(tee_response_payload.encode()).decode(),
        })

        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.text = tee_response

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.get = AsyncMock()
        tee._http = mock_http

        items = [
            {"prompt": "p1", "model_output": "o1", "category": "hate"},
            {"prompt": "p2", "model_output": "o2", "category": "pii_ip"},
        ]
        results = await tee.classify_batch(items)

        assert len(results) == 2
        assert all(r["tee_attested"] for r in results)

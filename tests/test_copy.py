"""Compliant copy: templates render, policy judge blocks abuse."""
from punar.core.copy import TEMPLATES, generate_copy, policy_judge, select_template


def test_templates_exist_for_primary_interventions():
    for iv in ["silent_retry_aligned", "whatsapp_nudge_payment_link",
               "email_payment_link", "voice_call", "escalate_manual"]:
        assert iv in TEMPLATES, iv


def test_approved_copy_passes_judge():
    text, ok, violations = generate_copy("whatsapp_nudge_payment_link",
                                         "insufficient_funds", "en",
                                         merchant="Acme", amount="2499",
                                         link="https://example.com/pay")
    assert ok is True, violations
    assert "2499" in text and "Acme" in text and "https://example.com/pay" in text


def test_multilingual_rendering():
    for lang in ("en", "hi", "hinglish"):
        text, ok, v = generate_copy("email_payment_link", "expired_card", lang,
                                    merchant="X", amount="999", link="l")
        assert ok is True, (lang, v)
        assert "999" in text


def test_judge_blocks_threats_and_shaming():
    ok, v = policy_judge("Pay now or we take legal action, defaulter")
    assert ok is False and len(v) >= 1


def test_judge_blocks_default_and_seizure_language():
    for bad in ["defaulting on payment", "we will seize assets", "we will sue you"]:
        ok, v = policy_judge(bad)
        assert ok is False, bad


def test_judge_blocks_excessive_caps_and_punctuation():
    ok, v = policy_judge("PAY NOW !!! IMMEDIATE ACTION REQUIRED !!!")
    assert ok is False


def test_judge_blocks_unapproved_discounts():
    ok, v = policy_judge("Pay now and get 50% flat discount waived")
    assert ok is False


def test_clean_copy_passes():
    ok, v = policy_judge("Your payment of INR 499 could not be processed. Retry: {link}")
    assert ok is True, v


def test_unknown_intervention_fallback():
    text = select_template("brand_new_action_42", "bank_decline_general", "en")
    out, ok, v = generate_copy("brand_new_action_42", "bank_decline_general", "en",
                               amount="100", link="l")
    assert isinstance(text, str) and len(text) > 10 and ok is True

"""Compliant EN/Hindi/Hinglish message templates + deterministic policy judge.

Every outbound message is rendered from an approved template and passes a
policy judge (no threats, shaming, third-party disclosure, false urgency or
unapproved discounts) before it leaves Punar -- an RBI Fair-Practices /
digital-lending safeguard. An LLM-based judge can be dropped in later; the
verdict always lands in audit.

The judge is MULTILINGUAL on purpose: two-thirds of the shipped templates are
Hindi/Hinglish, so an English-only rule set would wave through exactly the
copy that is most likely to breach RBI recovery-conduct rules. Every category
below (legal threat, shaming, third-party disclosure, harassment, unapproved
discount) is expressed in English, romanized Hinglish AND Devanagari.

Input is Unicode-normalized (NFKC) and curly quotes are folded to ASCII before
matching, so "won't" cannot slip past a rule written with a straight quote.
"""
import re
import unicodedata
from typing import Any

from punar.core.taxonomy import get_reason

DEFAULTS = {"brand": "Punar", "sender": "Punar Payments",
            "optout": "Reply STOP to opt out of payment reminders."}

TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    "silent_retry_aligned": {
        "en": "We'll automatically retry your {merchant} payment of INR {amount} tonight after 7 PM IST. No action needed. {optout}",
        "hi": "Hum aaj raat 7 baje ke baad apne aap {merchant} ka INR {amount} ka bhugtan dobara prayas karenge. Koi kaarrwai zaroori nahin. {optout}",
        "hinglish": "Hum aaj raat 7 PM IST ke baad automatically {merchant} ka INR {amount} wala payment retry kar denge. Aapko kuch nahi karna. {optout}",
    },
    "whatsapp_nudge_payment_link": {
        "en": "Hi from {brand} for {merchant}: your payment of INR {amount} could not be processed. Retry securely in one tap: {link}. Need help? Reply HELP. {optout}",
        "hi": "{brand} se {merchant} ke liye: INR {amount} ka bhugtan pura nahi ho paaya. Ek tap mein surakshit tareeke se retry karein: {link}. Madad chahiye to HELP bhejein. {optout}",
        "hinglish": "{merchant} ka INR {amount} payment process nahi ho paaya ({brand}). Yahan click karke turant retry karein: {link}. Koi doubt ho to reply karein. {optout}",
    },
    "email_payment_link": {
        "en": "Subject: Complete your {merchant} payment (INR {amount})\nBody: Hello, your recurring payment failed. You can complete it safely here: {link}. If your card expired, update it at the same link. Regards, {sender}. {optout}",
        "hi": "Subject: Apna {merchant} bhugtan pura karein (INR {amount})\nBody: Namaste, aapka aavartik bhugtan viphal raha. Ise yahan surakshit roop se pura karein: {link}. Dhanyavaad, {sender}. {optout}",
        "hinglish": "Subject: Apna {merchant} payment complete karein (INR {amount})\nBody: Aapka recurring payment fail ho gaya. Yahan se safely complete karein: {link}. Thanks, {sender}. {optout}",
    },
    "voice_call": {
        "en": "Script: Namaste, this is {brand} calling for {merchant}. Your payment of INR {amount} did not go through. Please complete it using the link sent to you, or speak to an agent. We never ask for OTP or PIN. Thank you.",
        "hi": "Script: Namaste, yeh {brand} bol raha hai, {merchant} ki taraf se. Aapka INR {amount} ka bhugtan pura nahi hua. Kripya bheje gaye link se pura karein ya agent se baat karein. Hum kabhi OTP ya PIN nahin maangte. Dhanyavaad.",
    },
    "payment_link_sms": {
        "en": "{merchant}: INR {amount} payment failed. Retry here: {link}. Never share OTP/PIN. STOP to opt out.",
        "hi": "{merchant}: INR {amount} bhugtan na hua. Yahan retry karein: {link}. OTP/PIN kisi ko na dein. STOP likh kar opt-out karein.",
    },
    "promise_to_pay": {
        "en": "No problem. Tell us a date and we'll retry on that day for {merchant} (INR {amount}): {link}. {optout}",
        "hi": "Koi baat nahin. Bas ek taareekh batayein, hum us din {merchant} (INR {amount}) ka retry kar denge: {link}. {optout}",
    },
    "escalate_manual": {
        "en": "Internal note: routed to human review for reason={reason}. Do not auto-contact; verify mandate status and customer preferences before outreach.",
        "hi": "Internal note: manual review ko bheja gaya, kaaran={reason}. Swaal se sampark na karein; pehle mandate aur customer pasand jaanch lein.",
    },
}

# ---------------------------------------------------------------------------
# Policy judge rule set. Each entry is (violation_code, compiled_pattern).
# Codes are semantic (not positional indexes) so an audit reviewer can read
# them, and so reordering the list never renumbers a historical violation.
# ---------------------------------------------------------------------------
_RULES: list[tuple[str, str]] = [
    # --- legal threats -----------------------------------------------------
    ("legal_threat", r"\blegal\s+action\b|\bcourt\b|\bpolice\b|\barrest(ed|ing)?\b"
                     r"|\bcriminal\b|\bsue\b|\blawsuit\b|\bprosecut\w*\b"
                     r"|\bsection\s+138\b|\brecovery\s+agent\b"),
    # Hinglish/romanized legal threats: kanooni/qanooni karyavahi, adalat, mukadma.
    ("legal_threat", r"\b(kanoon\w*|kanuni\w*|qanoon\w*|qanuni\w*|adaalat\w*|adalat\w*"
                     r"|mukadma\w*|muqadma\w*|mukadama\w*|vakil|wakil|giraftar\w*"
                     r"|jail\s+bhej\w*|police\s+(mein|me)\s+(complaint|shikayat))\b"),
    # Devanagari legal threats.
    ("legal_threat", r"(कानूनी|क़ानूनी|अदालत|न्यायालय|पुलिस|मुकदम|गिरफ्तार|वकील|जेल)"),

    # --- shaming / defamation ---------------------------------------------
    ("shaming", r"\bdefaulter\b|\bfraudster\b|\bshame\b|\bshameful\b|\bdisgrace\b"
                r"|\bblacklist(ed|ing)?\b|\bname\s+and\s+shame\b"),
    ("shaming", r"\b(sharm|sharam|besharam|beizzat\w*|be\s*izzat\w*|badnaam\w*"
                r"|badnami|zaleel|zillat|chor\b|chori\s+kar)\w*"),
    ("shaming", r"(शर्म|बेइज्ज|बदनाम|ज़लील|जलील|चोर)"),

    # --- third-party disclosure (RBI: never contact family/employer/contacts)
    ("third_party_disclosure", r"(?=[\s\S]*(\b(family|relatives?|neighbou?rs?|employer|boss"
                               r"|colleagues?|friends)\b|\byour\s+contacts?\b"
                               r"|\bcontacts?\s+list\b|\bphone\s*book\b))"
                               r"(?=[\s\S]*\b(tell|inform|notify|call|contact|disclose|share)\w*\b)"),
    ("third_party_disclosure", r"(?=[\s\S]*\b(gharwal\w+|ghar\s*wal\w+|ghar\s*wale|parivar\w*"
                               r"|padosi\w*|rishtedar\w*|dost\w*|office\s+wal\w+|boss)\b)"
                               r"(?=[\s\S]*\b(bata\w*|batayen\w*|bataeng\w*|batana|inform\w*"
                               r"|sampark|phone\s+kar\w*|call\s+kar\w*|shikayat)\w*)"),
    ("third_party_disclosure", r"(?=[\s\S]*(घरवाल|घर\s*वाल|परिवार|पड़ोसी|रिश्तेदार|दोस्तों|ऑफिस))"
                               r"(?=[\s\S]*(बताएंगे|बता\s*देंगे|बतायेंगे|सूचित|संपर्क|फोन))"),

    # --- harassment / coercion --------------------------------------------
    ("harassment", r"\bthreat(en|s|ening|ened)?\b|\bharass(ment|ing|ed)?\b"
                   r"|\bwe\s+will\s+not\s+stop\b|\bwe\s+will\s+keep\s+calling\b"),
    ("harassment", r"\b(dhamki\w*|dhamka\w*|dhamkay\w*|pareshan\s+kar\w*|zabardasti"
                   r"|majboor\s+kar\w*|dara\s*kar\w*|darayen\w*)\b"),
    ("harassment", r"(धमकी|धमका|परेशान\s*कर|ज़बरदस्ती|जबरदस्ती|मजबूर\s*कर)"),

    # --- asset seizure -----------------------------------------------------
    # `attach` only in the legal attachment-of-property sense; "we attach the
    # invoice" is innocuous and must NOT trip the judge.
    ("asset_seizure", r"\bseize\b|\bseizure\b|\bconfiscat\w*\b|\brepossess\w*\b"
                      r"|\battach(ment)?\s+(of\s+)?(your\s+|the\s+)?"
                      r"(property|assets?|salary|wages|bank\s+account|account)\b"),
    ("asset_seizure", r"\b(jabti|zabti|kurki|kurqi|sampatti\s+(jabt|zabt))\w*\b"),

    # --- default / delinquency labelling ----------------------------------
    ("default_labelling", r"\bdefaulting\b|\bdefaulted\b|\byou\s+will\s+default\b"
                          r"|\bcredit\s+score\s+will\s+be\s+(destroyed|ruined)\b"),
    ("default_labelling", r"\b(defaulter\s+ghoshit|credit\s+score\s+kharab\s+kar)\w*\b"),

    # --- false urgency / capability threats --------------------------------
    ("false_urgency", r"\byou\s+won't\s+be\s+able\s+to\b|\byou\s+will\s+lose\s+access\b"
                      r"|\blast\s+and\s+final\s+warning\b|\bfinal\s+warning\b"),
    ("false_urgency", r"\b(aakhri\s+chetavni|antim\s+chetavni|warna\s+account\s+band)\b"),
    ("false_urgency", r"(आख़िरी\s*चेतावनी|अंतिम\s*चेतावनी|चेतावनी)"),
]

# Discounts must come from merchant-configured campaigns, never ad-hoc copy.
# Guarded so "toll-free helpline" and "free trial ends" do not read as an offer.
_DISCOUNT_RULES: list[tuple[str, str]] = [
    ("unapproved_discount", r"\b(discount|waiver|waived?|waive|cashback|rebate)\b"
                            r"|\b\d{1,3}\s*%\s*(off|discount|waiver)?\b"
                            r"|\bflat\s+(\d|inr|rs)\b"
                            r"|(?<![\w-])free\s+(month|months|trial|for\s+you)\b"
                            r"|\bspecial\s+offer\b|\blimited\s+time\s+offer\b"),
    ("unapproved_discount", r"\b(muft|chhoot|chhut|maafi|maaf\s+kar\w*|discount\s+de)\w*\b"),
    ("unapproved_discount", r"(मुफ़्त|मुफ्त|छूट|माफ़ी|माफी)"),
]

_COMPILED = [(code, re.compile(pat, re.IGNORECASE | re.UNICODE)) for code, pat in _RULES]
_COMPILED_DISCOUNT = [(code, re.compile(pat, re.IGNORECASE | re.UNICODE))
                      for code, pat in _DISCOUNT_RULES]

# Typographic characters that would otherwise defeat an ASCII-quoted pattern.
_QUOTE_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "′": "'", "ʼ": "'",
    "–": "-", "—": "-", "−": "-", " ": " ",
})


def normalize(text: str) -> str:
    """NFKC-normalize and fold typographic punctuation to ASCII for matching."""
    return unicodedata.normalize("NFKC", str(text)).translate(_QUOTE_FOLD)


def _pick_language(lang: str, reason: str) -> str:
    if lang in ("en", "hi", "hinglish"):
        return lang
    # Fallback mapping: prefer Hinglish for WhatsApp/SMS-style reasons, English otherwise.
    meta = get_reason(reason)
    if meta.requires_customer_action:
        return "hi"
    return "en"


def select_template(intervention: str, reason: str, lang: str = "") -> str:
    by_reason = TEMPLATES.get(intervention)
    if not by_reason:
        return f"{DEFAULTS['brand']}: please complete your {reason.replace('_', ' ')} payment (INR {{amount}}): {{link}}. {DEFAULTS['optout']}"
    lang = _pick_language(lang, reason)
    return by_reason.get(lang, next(iter(by_reason.values())))


def render(template: str, **kwargs: Any) -> str:
    ctx = dict(DEFAULTS)
    ctx.update({k: v for k, v in kwargs.items() if v is not None})
    return template.format(**ctx)


def policy_judge(text: str) -> tuple[bool, list[str]]:
    """Deterministic multilingual compliance check -> (allowed, violation_codes)."""
    norm = normalize(text)
    violations: list[str] = []
    for code, pat in _COMPILED + _COMPILED_DISCOUNT:
        if pat.search(norm) and code not in violations:
            violations.append(code)
    alphas = [ch for ch in norm if ch.isalpha()]
    if alphas and sum(1 for ch in alphas if ch.isupper()) / len(alphas) > 0.4:
        violations.append("excessive_capitalization")
    if norm.count("!") > 2:
        violations.append("excessive_punctuation")
    return len(violations) == 0, violations


def copy_defaults(policy: dict[str, Any] | None = None) -> dict[str, str]:
    """Merchant-configurable branding pulled from policy.json -> copy.*.

    Without this, every merchant's message would be branded with the hardcoded
    DEFAULTS, which is exactly the bug the config block was meant to prevent.
    """
    cfg = ((policy or {}).get("copy", {}) or {})
    return {
        "brand": str(cfg.get("brand_name") or DEFAULTS["brand"]),
        "sender": str(cfg.get("sender") or DEFAULTS["sender"]),
        "optout": str(cfg.get("optout_line") or DEFAULTS["optout"]),
    }


def generate_copy(intervention: str, reason: str, lang: str,
                  **kwargs: Any) -> tuple[str, bool, list[str]]:
    """Render approved copy and run it through the policy judge.

    Pass `policy=<policy dict>` to brand the message from policy.json; explicit
    brand/sender/optout kwargs still win over it.
    """
    policy = kwargs.pop("policy", None)
    template = select_template(intervention, reason, lang)
    ctx = copy_defaults(policy)
    ctx.update({k: v for k, v in kwargs.items() if v is not None})
    rendered = render(template, reason=reason, **ctx)
    allowed, violations = policy_judge(rendered)
    return rendered, allowed, violations

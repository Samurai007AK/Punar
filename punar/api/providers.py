"""Provider interfaces for everything Punar does NOT actually do yet.

Punar has **no message delivery integration**. There is no HTTP client, no
WhatsApp/SMS/email/voice vendor, no Razorpay SDK call anywhere in this
repository. The agent state machine renders copy and records a touch; nothing
leaves the process.

Rather than hide that behind a hard-coded ``"delivered": True``, this module
defines the four seams a production deployment must fill, and ships an
explicitly-labelled stub behind each one:

===========================  =====================================  ==========
Interface                    Production implementation              Ships as
===========================  =====================================  ==========
:class:`ChannelSender`       WhatsApp BSP / SMS / ESP / voice API    stub
:class:`ConsentLookup`       merchant CRM / DND + opt-out registry   stub
:class:`CustomerProfile...`  merchant customer directory             stub
:class:`OutcomeObserver`     ``payment.captured`` webhook stream     stub
===========================  =====================================  ==========

Every stub sets ``simulated=True`` on its result, and the API + audit record
propagate that flag verbatim. A message that was never sent is reported as
``delivered=False, simulated=True`` -- never as delivered.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "OutboundMessage",
    "DeliveryReceipt",
    "CustomerProfile",
    "ChannelSender",
    "ConsentLookup",
    "CustomerProfileLookup",
    "OutcomeObserver",
    "StubChannelSender",
    "StubConsentLookup",
    "StubCustomerProfileLookup",
    "StubOutcomeObserver",
    "ProviderRegistry",
    "SIMULATION_NOTICE",
]

SIMULATION_NOTICE = (
    "SIMULATED: Punar has no message-delivery integration wired up. No message "
    "was transmitted to any customer. Implement punar.api.providers.ChannelSender "
    "against a real provider before using this in production."
)


@dataclass(frozen=True)
class OutboundMessage:
    """One rendered, policy-judged message the agent wants to send."""

    case_id: str
    customer_id: str | None
    channel: str
    intervention: str
    body: str
    language: str = "en"
    payment_link: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeliveryReceipt:
    """The provider's answer. ``simulated`` is never inferred -- it is declared."""

    channel: str
    delivered: bool
    simulated: bool
    provider: str
    provider_message_id: str | None = None
    status: str = "not_sent"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "delivered": self.delivered,
            "simulated": self.simulated,
            "provider": self.provider,
            "provider_message_id": self.provider_message_id,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CustomerProfile:
    """Contact + preference data a real deployment must fetch per customer."""

    customer_id: str
    language: str = "en"
    phone: str | None = None
    email: str | None = None
    name: str | None = None
    source: str = "stub"


@runtime_checkable
class ChannelSender(Protocol):
    """Transmits one rendered message over one channel."""

    provider_name: str
    simulated: bool

    def supports(self, channel: str) -> bool: ...

    def send(self, message: OutboundMessage) -> DeliveryReceipt: ...


@runtime_checkable
class ConsentLookup(Protocol):
    """Authoritative answer to 'may we contact this customer at all?'."""

    provider_name: str
    simulated: bool

    def is_opted_out(self, customer_id: str | None) -> bool: ...


@runtime_checkable
class CustomerProfileLookup(Protocol):
    """Fetches contact details and language preference for a customer."""

    provider_name: str
    simulated: bool

    def fetch(self, customer_id: str | None) -> CustomerProfile | None: ...


@runtime_checkable
class OutcomeObserver(Protocol):
    """Reports whether an intervention actually recovered the payment.

    In production this is fed by the ``payment.captured`` / ``payment.paid``
    webhook stream, correlated back to the outreach that preceded it. It is
    inherently asynchronous; the synchronous signature here exists so the
    existing agent loop can be driven end to end.
    """

    provider_name: str
    simulated: bool

    def observe(self, case: dict[str, Any], intervention: str,
                now: datetime) -> bool | None: ...


# ------------------------------------------------------------------- stubs
class StubChannelSender:
    """Sends nothing. Records what *would* have been sent, honestly labelled."""

    provider_name = "stub"
    simulated = True

    def __init__(self) -> None:
        self.sent: list[OutboundMessage] = []

    def supports(self, channel: str) -> bool:
        return True

    def send(self, message: OutboundMessage) -> DeliveryReceipt:
        self.sent.append(message)
        digest = hashlib.sha256(
            f"{message.case_id}|{message.intervention}|{len(self.sent)}".encode()
        ).hexdigest()[:16]
        return DeliveryReceipt(
            channel=message.channel,
            delivered=False,
            simulated=True,
            provider=self.provider_name,
            provider_message_id=f"stub_{digest}",
            status="not_sent",
            detail=SIMULATION_NOTICE,
        )


class StubConsentLookup:
    """Consent registry backed by a static list from configuration.

    This is a real seam, not a no-op: ``PUNAR_STUB_OPTED_OUT_CUSTOMERS`` lets
    the opt-out guarantee be exercised end to end, and the API path now calls
    this before every run instead of hard-coding ``opted_out=False``.
    """

    provider_name = "stub"
    simulated = True

    def __init__(self, opted_out: Sequence[str] = ()) -> None:
        self._opted_out = {str(c) for c in opted_out}

    def is_opted_out(self, customer_id: str | None) -> bool:
        return bool(customer_id) and str(customer_id) in self._opted_out


class StubCustomerProfileLookup:
    """Returns no contact details -- because none are available in this repo."""

    provider_name = "stub"
    simulated = True

    def __init__(self, default_language: str = "en") -> None:
        self.default_language = default_language

    def fetch(self, customer_id: str | None) -> CustomerProfile | None:
        if not customer_id:
            return None
        return CustomerProfile(customer_id=str(customer_id),
                               language=self.default_language,
                               phone=None, email=None, name=None, source="stub")


class StubOutcomeObserver:
    """Deterministic pseudo-outcome so the loop is observable offline.

    NOT a model, NOT merchant-calibrated: a seeded coin flip at
    ``success_rate``. Replace with the ``payment.captured`` correlation before
    quoting any recovery number as real.
    """

    provider_name = "stub"
    simulated = True

    def __init__(self, success_rate: float = 0.6, seed: str = "punar") -> None:
        self.success_rate = float(success_rate)
        self.seed = seed

    def observe(self, case: dict[str, Any], intervention: str,
                now: datetime) -> bool | None:
        case_id = str(case.get("case_id", ""))
        key = int.from_bytes(
            hashlib.sha256(f"{self.seed}|{case_id}|{intervention}".encode()).digest()[:8],
            "big")
        return random.Random(key).random() < self.success_rate


@dataclass
class ProviderRegistry:
    """The set of external systems the service talks to (all stubs by default)."""

    sender: ChannelSender = field(default_factory=StubChannelSender)
    consent: ConsentLookup = field(default_factory=StubConsentLookup)
    profiles: CustomerProfileLookup = field(default_factory=StubCustomerProfileLookup)
    outcomes: OutcomeObserver = field(default_factory=StubOutcomeObserver)

    @property
    def fully_simulated(self) -> bool:
        return all(getattr(p, "simulated", True)
                   for p in (self.sender, self.consent, self.profiles, self.outcomes))

    def describe(self) -> dict[str, Any]:
        """Machine-readable provenance, surfaced on /health and in every case."""
        return {
            "message_delivery": {
                "provider": getattr(self.sender, "provider_name", "unknown"),
                "simulated": bool(getattr(self.sender, "simulated", True)),
            },
            "consent_lookup": {
                "provider": getattr(self.consent, "provider_name", "unknown"),
                "simulated": bool(getattr(self.consent, "simulated", True)),
            },
            "customer_profile": {
                "provider": getattr(self.profiles, "provider_name", "unknown"),
                "simulated": bool(getattr(self.profiles, "simulated", True)),
            },
            "outcome_observation": {
                "provider": getattr(self.outcomes, "provider_name", "unknown"),
                "simulated": bool(getattr(self.outcomes, "simulated", True)),
            },
            "fully_simulated": self.fully_simulated,
            "notice": SIMULATION_NOTICE if self.fully_simulated else "",
        }

    @classmethod
    def from_settings(cls, settings: Any) -> ProviderRegistry:
        return cls(
            sender=StubChannelSender(),
            consent=StubConsentLookup(settings.stub_opted_out_customers),
            profiles=StubCustomerProfileLookup(),
            outcomes=StubOutcomeObserver(success_rate=settings.outcome_simulation_rate),
        )

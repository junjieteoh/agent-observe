"""
PII Handling Example

Demonstrates how to automatically detect and redact/hash/tokenize
Personally Identifiable Information (PII) before it's stored.

Use cases:
- GDPR/CCPA compliance
- HIPAA compliance for healthcare
- PCI DSS compliance for payment data
- General data privacy protection
"""

from agent_observe import observe, tool, PIIConfig
from agent_observe.config import Config, CaptureMode, Environment


# =============================================================================
# EXAMPLE 1: Basic PII Redaction
# =============================================================================

def example_redaction():
    """Demonstrate basic PII redaction."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic PII Redaction")
    print("="*60)

    observe.install(
        config=Config(mode=CaptureMode.FULL, env=Environment.DEV),
        pii=PIIConfig(
            enabled=True,
            action="redact",  # Replace with [EMAIL_REDACTED], etc.
            patterns={
                "email": True,
                "phone": True,
                "ssn": True,
            },
        )
    )

    @tool(name="process_customer", kind="generic")
    def process_customer(name: str, email: str, phone: str) -> dict:
        """Process customer information."""
        return {
            "name": name,
            "email": email,
            "phone": phone,
            "status": "processed",
        }

    with observe.run("pii-redaction-demo") as run:
        result = process_customer(
            name="John Doe",
            email="john.doe@example.com",
            phone="555-123-4567",
        )
        print(f"Function returned: {result}")

    observe.sink.flush()

    # The stored data will have PII redacted
    print("\nStored data has PII redacted:")
    print("  email -> [EMAIL_REDACTED]")
    print("  phone -> [PHONE_REDACTED]")


# =============================================================================
# EXAMPLE 2: PII Hashing (Consistent, One-Way)
# =============================================================================

def example_hashing():
    """Demonstrate PII hashing for analytics without exposure."""
    print("\n" + "="*60)
    print("EXAMPLE 2: PII Hashing (for analytics)")
    print("="*60)

    observe._cleanup()  # Reset for new config

    observe.install(
        config=Config(mode=CaptureMode.FULL, env=Environment.DEV),
        pii=PIIConfig(
            enabled=True,
            action="hash",  # Replace with consistent hash
            hash_salt="my-secret-salt",  # Ensures consistent hashes
            patterns={
                "email": True,
            },
        )
    )

    @tool(name="log_activity", kind="generic")
    def log_activity(user_email: str, action: str) -> dict:
        return {"user": user_email, "action": action}

    with observe.run("pii-hash-demo") as run:
        # Same email will always produce the same hash
        log_activity("jane@example.com", "login")
        log_activity("jane@example.com", "purchase")
        log_activity("bob@example.com", "login")

    observe.sink.flush()

    print("\nHashed emails enable:")
    print("  - Counting unique users without storing emails")
    print("  - Tracking user journeys without PII exposure")
    print("  - Consistent hashes across runs (with same salt)")


# =============================================================================
# EXAMPLE 3: PII Tokenization (Reversible)
# =============================================================================

def example_tokenization():
    """Demonstrate reversible PII tokenization."""
    print("\n" + "="*60)
    print("EXAMPLE 3: PII Tokenization (Reversible)")
    print("="*60)

    observe._cleanup()

    pii_config = PIIConfig(
        enabled=True,
        action="tokenize",  # Replace with reversible token
        patterns={
            "email": True,
            "ssn": True,
        },
    )

    observe.install(
        config=Config(mode=CaptureMode.FULL, env=Environment.DEV),
        pii=pii_config,
    )

    @tool(name="verify_identity", kind="generic")
    def verify_identity(email: str, ssn: str) -> dict:
        return {"email": email, "ssn": ssn, "verified": True}

    with observe.run("pii-tokenize-demo") as run:
        verify_identity("sensitive@example.com", "123-45-6789")

    observe.sink.flush()

    # Tokens can be reversed if needed (by authorized personnel)
    print("\nTokenization enables:")
    print("  - Storing tokens instead of PII")
    print("  - Authorized recovery of original values")
    print("  - Audit trail without exposure")


# =============================================================================
# EXAMPLE 4: Custom PII Patterns
# =============================================================================

def example_custom_patterns():
    """Demonstrate custom PII detection patterns."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom PII Patterns")
    print("="*60)

    observe._cleanup()

    observe.install(
        config=Config(mode=CaptureMode.FULL, env=Environment.DEV),
        pii=PIIConfig(
            enabled=True,
            action="redact",
            patterns={
                # Built-in patterns
                "email": True,
                "phone": True,
                # Custom patterns for your organization
                "employee_id": r"EMP-\d{6}",           # EMP-123456
                "order_number": r"ORD-[A-Z]{2}\d{8}",  # ORD-AB12345678
                "api_key": r"sk-[a-zA-Z0-9]{32}",      # sk-xxxx...
                "internal_ip": r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # 10.x.x.x
            },
        )
    )

    @tool(name="process_order", kind="generic")
    def process_order(order_id: str, employee: str, api_key: str) -> dict:
        return {
            "order": order_id,
            "processed_by": employee,
            "api_used": api_key,
        }

    with observe.run("custom-pii-demo") as run:
        process_order(
            order_id="ORD-AB12345678",
            employee="EMP-789012",
            api_key="sk-abcdefghijklmnopqrstuvwxyz123456",
        )

    observe.sink.flush()

    print("\nCustom patterns detected and redacted:")
    print("  ORD-AB12345678 -> [ORDER_NUMBER_REDACTED]")
    print("  EMP-789012 -> [EMPLOYEE_ID_REDACTED]")
    print("  sk-... -> [API_KEY_REDACTED]")


# =============================================================================
# EXAMPLE 5: PII with Detection Callback
# =============================================================================

def example_detection_callback():
    """Demonstrate PII detection with alerting."""
    print("\n" + "="*60)
    print("EXAMPLE 5: PII Detection with Alerting")
    print("="*60)

    observe._cleanup()

    detected_pii = []

    def on_pii_detected(pattern_name: str, original_value: str, replacement: str):
        """Callback when PII is detected - can send alerts, log, etc."""
        detected_pii.append({
            "pattern": pattern_name,
            "original_preview": original_value[:3] + "***",  # Safe preview
            "replacement": replacement,
        })
        print(f"  ALERT: Detected {pattern_name} in data!")

    observe.install(
        config=Config(mode=CaptureMode.FULL, env=Environment.DEV),
        pii=PIIConfig(
            enabled=True,
            action="redact",
            patterns={
                "email": True,
                "ssn": True,
                "credit_card": True,
            },
            on_detect=on_pii_detected,
        )
    )

    @tool(name="handle_payment", kind="generic")
    def handle_payment(email: str, card: str, ssn: str) -> dict:
        return {"email": email, "card": card, "ssn": ssn}

    with observe.run("pii-alert-demo") as run:
        handle_payment(
            email="customer@example.com",
            card="4111-1111-1111-1111",
            ssn="123-45-6789",
        )

    observe.sink.flush()

    print(f"\nTotal PII instances detected: {len(detected_pii)}")
    for item in detected_pii:
        print(f"  - {item['pattern']}: {item['original_preview']}")


# =============================================================================
# EXAMPLE 6: PII Flag Mode (Keep but Mark)
# =============================================================================

def example_flag_mode():
    """Demonstrate PII flagging without removal."""
    print("\n" + "="*60)
    print("EXAMPLE 6: PII Flag Mode (Mark but Keep)")
    print("="*60)

    observe._cleanup()

    observe.install(
        config=Config(mode=CaptureMode.FULL, env=Environment.DEV),
        pii=PIIConfig(
            enabled=True,
            action="flag",  # Keep original but mark as PII
            patterns={
                "email": True,
            },
        )
    )

    @tool(name="send_notification", kind="generic")
    def send_notification(recipient: str, message: str) -> dict:
        return {"to": recipient, "sent": True}

    with observe.run("pii-flag-demo") as run:
        send_notification("user@example.com", "Hello!")

    observe.sink.flush()

    print("\nFlag mode output:")
    print("  user@example.com -> [PII:email]user@example.com[/PII]")
    print("\nUse case: Internal systems where you need the data")
    print("but want to clearly mark PII for data governance.")


# =============================================================================
# Run all examples
# =============================================================================

if __name__ == "__main__":
    example_redaction()
    example_hashing()
    example_tokenization()
    example_custom_patterns()
    example_detection_callback()
    example_flag_mode()

    print("\n" + "="*60)
    print("All PII examples completed!")
    print("="*60)

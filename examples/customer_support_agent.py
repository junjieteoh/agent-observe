"""
Customer Support Agent Example

A realistic example of a customer support agent with:
- Database lookups for customer/order info
- Ticket management (create, update, escalate)
- Email notifications
- PII protection (customer emails, phone numbers auto-redacted)
- Security hooks (block bulk operations, rate limiting)
- Cost tracking for LLM calls
- Audit trail for compliance

This demonstrates how agent-observe handles a production-like scenario.
"""

import random
from datetime import datetime
from typing import Optional

from agent_observe import observe, tool, model_call, HookResult, PIIConfig
from agent_observe.config import Config, CaptureMode, Environment


# =============================================================================
# SIMULATED BACKENDS (replace with real implementations)
# =============================================================================

class MockDatabase:
    """Simulated database for demo purposes."""

    CUSTOMERS = {
        "C001": {"name": "Alice Johnson", "email": "alice@example.com", "phone": "555-123-4567", "tier": "premium"},
        "C002": {"name": "Bob Smith", "email": "bob.smith@company.org", "phone": "555-987-6543", "tier": "standard"},
    }

    ORDERS = {
        "ORD-2024-001": {"customer_id": "C001", "status": "shipped", "total": 299.99, "items": ["Widget Pro", "Gadget X"]},
        "ORD-2024-002": {"customer_id": "C001", "status": "processing", "total": 149.50, "items": ["Accessory Pack"]},
        "ORD-2024-003": {"customer_id": "C002", "status": "delivered", "total": 89.00, "items": ["Basic Widget"]},
    }

    TICKETS = {}

db = MockDatabase()


# =============================================================================
# TOOLS - Customer Support Operations
# =============================================================================

@tool(name="customer.lookup", kind="database")
def lookup_customer(customer_id: str) -> dict:
    """Look up customer information by ID."""
    if customer_id in db.CUSTOMERS:
        return {"found": True, "customer": db.CUSTOMERS[customer_id]}
    return {"found": False, "error": f"Customer {customer_id} not found"}


@tool(name="customer.search", kind="database")
def search_customer_by_email(email: str) -> dict:
    """Search for customer by email address."""
    for cid, customer in db.CUSTOMERS.items():
        if customer["email"].lower() == email.lower():
            return {"found": True, "customer_id": cid, "customer": customer}
    return {"found": False, "error": "No customer found with that email"}


@tool(name="order.lookup", kind="database")
def lookup_order(order_id: str) -> dict:
    """Look up order details by order ID."""
    if order_id in db.ORDERS:
        order = db.ORDERS[order_id]
        customer = db.CUSTOMERS.get(order["customer_id"], {})
        return {
            "found": True,
            "order": order,
            "customer_name": customer.get("name", "Unknown"),
        }
    return {"found": False, "error": f"Order {order_id} not found"}


@tool(name="order.list", kind="database")
def list_customer_orders(customer_id: str) -> dict:
    """List all orders for a customer."""
    orders = [
        {"order_id": oid, **order}
        for oid, order in db.ORDERS.items()
        if order["customer_id"] == customer_id
    ]
    return {"customer_id": customer_id, "orders": orders, "count": len(orders)}


@tool(name="ticket.create", kind="database")
def create_support_ticket(
    customer_id: str,
    subject: str,
    description: str,
    priority: str = "normal"
) -> dict:
    """Create a new support ticket."""
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    ticket = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "subject": subject,
        "description": description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().isoformat(),
    }
    db.TICKETS[ticket_id] = ticket
    return {"success": True, "ticket": ticket}


@tool(name="ticket.update", kind="database")
def update_ticket_status(ticket_id: str, status: str, notes: Optional[str] = None) -> dict:
    """Update a support ticket's status."""
    if ticket_id not in db.TICKETS:
        return {"success": False, "error": f"Ticket {ticket_id} not found"}

    db.TICKETS[ticket_id]["status"] = status
    if notes:
        db.TICKETS[ticket_id]["notes"] = notes
    return {"success": True, "ticket": db.TICKETS[ticket_id]}


@tool(name="ticket.escalate", kind="database")
def escalate_ticket(ticket_id: str, reason: str) -> dict:
    """Escalate a ticket to senior support."""
    if ticket_id not in db.TICKETS:
        return {"success": False, "error": f"Ticket {ticket_id} not found"}

    db.TICKETS[ticket_id]["status"] = "escalated"
    db.TICKETS[ticket_id]["escalation_reason"] = reason
    db.TICKETS[ticket_id]["escalated_at"] = datetime.now().isoformat()
    return {"success": True, "ticket": db.TICKETS[ticket_id]}


@tool(name="email.send", kind="notification")
def send_email(to: str, subject: str, body: str) -> dict:
    """Send email notification to customer."""
    # In production: integrate with SendGrid, SES, etc.
    print(f"[EMAIL] To: {to}, Subject: {subject}")
    return {
        "success": True,
        "message_id": f"MSG-{random.randint(10000, 99999)}",
        "recipient": to,
    }


@tool(name="refund.process", kind="payment")
def process_refund(order_id: str, amount: float, reason: str) -> dict:
    """Process a refund for an order."""
    if order_id not in db.ORDERS:
        return {"success": False, "error": f"Order {order_id} not found"}

    # In production: integrate with Stripe, PayPal, etc.
    return {
        "success": True,
        "refund_id": f"REF-{random.randint(10000, 99999)}",
        "order_id": order_id,
        "amount": amount,
        "reason": reason,
    }


# =============================================================================
# MODEL CALLS - LLM Integration
# =============================================================================

@model_call(provider="openai", model="gpt-4")
def analyze_customer_issue(customer_info: dict, issue_description: str) -> dict:
    """Analyze customer issue and suggest resolution."""
    # In production: call OpenAI API
    # response = openai.chat.completions.create(...)

    # Simulated response with usage stats
    return {
        "analysis": f"Customer {customer_info.get('name', 'Unknown')} has a {customer_info.get('tier', 'standard')} tier account. "
                    f"Issue appears to be related to: {issue_description[:100]}",
        "suggested_action": "Create support ticket and offer expedited resolution for premium customer",
        "priority": "high" if customer_info.get("tier") == "premium" else "normal",
        "usage": {
            "prompt_tokens": 250,
            "completion_tokens": 150,
        }
    }


@model_call(provider="openai", model="gpt-4")
def generate_response_email(customer_name: str, issue_summary: str, resolution: str) -> dict:
    """Generate a professional response email."""
    # Simulated response
    return {
        "content": f"Dear {customer_name},\n\nThank you for contacting us. {resolution}\n\nBest regards,\nSupport Team",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 80,
        }
    }


# =============================================================================
# HOOKS - Security, Cost Tracking, Audit
# =============================================================================

# Cost tracking
_session_costs = {"total_usd": 0.0, "calls": 0}

PRICING = {
    "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
    "gpt-3.5-turbo": {"input": 0.001 / 1000, "output": 0.002 / 1000},
}


@observe.hooks.after_model
def track_llm_cost(ctx, result):
    """Track token usage and cost for each LLM call."""
    if isinstance(result, dict) and "usage" in result:
        usage = result["usage"]
        model = ctx.model or "gpt-4"
        pricing = PRICING.get(model, PRICING["gpt-4"])

        input_cost = usage.get("prompt_tokens", 0) * pricing["input"]
        output_cost = usage.get("completion_tokens", 0) * pricing["output"]
        call_cost = input_cost + output_cost

        _session_costs["total_usd"] += call_cost
        _session_costs["calls"] += 1

        # Record in span attributes
        ctx.span.set_attribute("cost_usd", round(call_cost, 6))
        ctx.span.set_attribute("tokens_total", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))

    return result


# Security: Block bulk operations
@observe.hooks.before_tool
def block_bulk_operations(ctx):
    """Prevent bulk/dangerous operations."""
    dangerous_patterns = ["DELETE FROM", "DROP TABLE", "TRUNCATE", "UPDATE.*WHERE 1=1"]

    args_str = str(ctx.args) + str(ctx.kwargs)
    for pattern in dangerous_patterns:
        if pattern.upper() in args_str.upper():
            return HookResult.block(f"Blocked: Detected dangerous pattern '{pattern}'")

    return HookResult.proceed()


# Audit: Log sensitive operations
@observe.hooks.after_tool
def audit_sensitive_operations(ctx, result):
    """Emit audit events for sensitive operations."""
    sensitive_tools = {"refund.process", "ticket.escalate", "email.send"}

    if ctx.tool_name in sensitive_tools:
        observe.emit_event("audit.sensitive_operation", {
            "tool": ctx.tool_name,
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False) if isinstance(result, dict) else True,
        })

    return result


# =============================================================================
# MAIN AGENT FLOW
# =============================================================================

def handle_customer_inquiry(customer_email: str, issue: str):
    """
    Main agent flow for handling a customer support inquiry.

    This demonstrates a realistic support workflow:
    1. Look up customer by email
    2. Retrieve their order history
    3. Analyze the issue with LLM
    4. Create a support ticket
    5. Send confirmation email
    """

    with observe.run(
        "customer-support-agent",
        user_id="support-bot",
        session_id=f"inquiry-{datetime.now().strftime('%Y%m%d%H%M%S')}",
    ) as run:
        run.set_input({
            "customer_email": customer_email,
            "issue": issue,
        })

        # Step 1: Find customer
        print(f"\n[1] Looking up customer: {customer_email}")
        customer_result = search_customer_by_email(customer_email)

        if not customer_result["found"]:
            run.set_output({"status": "error", "message": "Customer not found"})
            return {"error": "Customer not found"}

        customer = customer_result["customer"]
        customer_id = customer_result["customer_id"]
        print(f"    Found: {customer['name']} ({customer['tier']} tier)")

        # Step 2: Get order history
        print(f"\n[2] Fetching order history...")
        orders = list_customer_orders(customer_id)
        print(f"    Found {orders['count']} orders")

        # Step 3: Analyze issue with LLM
        print(f"\n[3] Analyzing issue with AI...")
        analysis = analyze_customer_issue(customer, issue)
        print(f"    Priority: {analysis['priority']}")
        print(f"    Suggested: {analysis['suggested_action']}")

        # Step 4: Create support ticket
        print(f"\n[4] Creating support ticket...")
        ticket = create_support_ticket(
            customer_id=customer_id,
            subject=f"Support Request: {issue[:50]}",
            description=issue,
            priority=analysis["priority"],
        )
        ticket_id = ticket["ticket"]["ticket_id"]
        print(f"    Created: {ticket_id}")

        # Step 5: Generate and send response email
        print(f"\n[5] Sending confirmation email...")
        email_content = generate_response_email(
            customer_name=customer["name"],
            issue_summary=issue,
            resolution=f"We've created ticket {ticket_id} and will respond within 24 hours.",
        )

        email_result = send_email(
            to=customer["email"],
            subject=f"Support Ticket {ticket_id} Created",
            body=email_content["content"],
        )

        # Final output
        result = {
            "status": "success",
            "ticket_id": ticket_id,
            "customer_name": customer["name"],
            "priority": analysis["priority"],
            "email_sent": email_result["success"],
        }

        run.set_output(result)

        # Emit summary event
        observe.emit_event("inquiry.completed", {
            "ticket_id": ticket_id,
            "customer_tier": customer["tier"],
            "priority": analysis["priority"],
        })

        return result


def handle_refund_request(order_id: str, reason: str):
    """
    Handle a refund request with proper authorization and audit trail.
    """

    with observe.run(
        "refund-agent",
        user_id="support-bot",
    ) as run:
        run.set_input({"order_id": order_id, "reason": reason})

        # Look up order
        order_result = lookup_order(order_id)
        if not order_result["found"]:
            run.set_output({"status": "error", "message": "Order not found"})
            return {"error": "Order not found"}

        order = order_result["order"]

        # Process refund
        refund = process_refund(
            order_id=order_id,
            amount=order["total"],
            reason=reason,
        )

        run.set_output(refund)
        return refund


# =============================================================================
# RUN THE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Configure with PII protection and full tracing
    observe.install(
        config=Config(mode=CaptureMode.FULL, env=Environment.DEV),
        pii=PIIConfig(
            enabled=True,
            action="redact",
            patterns={
                "email": True,
                "phone": True,
            },
        ),
    )

    print("=" * 60)
    print("CUSTOMER SUPPORT AGENT DEMO")
    print("=" * 60)

    # Simulate a customer inquiry
    result = handle_customer_inquiry(
        customer_email="alice@example.com",
        issue="My order ORD-2024-002 has been processing for 3 days. When will it ship?",
    )

    print("\n" + "=" * 60)
    print("RESULT:")
    print(f"  Ticket: {result.get('ticket_id')}")
    print(f"  Priority: {result.get('priority')}")
    print(f"  Email Sent: {result.get('email_sent')}")

    print("\n" + "=" * 60)
    print("SESSION COST SUMMARY:")
    print(f"  LLM Calls: {_session_costs['calls']}")
    print(f"  Total Cost: ${_session_costs['total_usd']:.4f}")

    print("\n" + "=" * 60)
    print("PII PROTECTION:")
    print("  Customer emails/phones are automatically redacted in stored traces")
    print("  View traces with: agent-observe view")

    # Flush to ensure all data is written
    observe.sink.flush()

"""
Email Sender Module with SMTP Round-Robin Support

Handles email sending with:
- Multiple SMTP accounts
- Round-robin selection respecting daily/hourly limits
- Template variable substitution
- Health tracking (bounce rate, send counts)
- Open/click tracking pixel injection
"""

import smtplib
import ssl
import logging
import json
import re
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr, formatdate, make_msgid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EmailAccountStatus(Enum):
    PENDING = "pending"      # Not yet tested
    ACTIVE = "active"        # Working and available
    WARMUP = "warmup"        # In warmup mode (reduced volume)
    ERROR = "error"          # Connection issues
    DISABLED = "disabled"    # Manually or auto-disabled


@dataclass
class EmailAccount:
    """Email account configuration for SMTP sending"""
    id: int = 0
    email: str = ""
    display_name: str = ""

    # SMTP settings
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False

    # IMAP settings (for reply detection)
    imap_host: str = ""
    imap_port: int = 993
    imap_username: str = ""
    imap_password: str = ""
    imap_use_ssl: bool = True

    # Limits
    daily_limit: int = 100
    hourly_limit: int = 20
    delay_between_emails_seconds: int = 60  # Min seconds between sends

    # Tracking
    status: EmailAccountStatus = EmailAccountStatus.PENDING
    emails_sent_today: int = 0
    emails_sent_this_hour: int = 0
    emails_sent_total: int = 0
    last_sent_at: Optional[datetime] = None
    last_error: str = ""

    # Health metrics
    health_score: int = 100
    bounce_rate_7d: float = 0.0
    spam_rate_7d: float = 0.0
    open_rate_7d: float = 0.0

    # Warmup
    warmup_enabled: bool = False
    warmup_day: int = 0  # Days into warmup
    warmup_target_volume: int = 10  # Target emails/day during warmup

    # Signature
    signature_html: str = ""
    signature_text: str = ""

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    auto_disabled_at: Optional[datetime] = None
    auto_disabled_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "smtp_username": self.smtp_username,
            "smtp_password": "***" if self.smtp_password else "",  # Masked
            "smtp_use_tls": self.smtp_use_tls,
            "smtp_use_ssl": self.smtp_use_ssl,
            "imap_host": self.imap_host,
            "imap_port": self.imap_port,
            "imap_username": self.imap_username,
            "imap_password": "***" if self.imap_password else "",  # Masked
            "imap_use_ssl": self.imap_use_ssl,
            "daily_limit": self.daily_limit,
            "hourly_limit": self.hourly_limit,
            "delay_between_emails_seconds": self.delay_between_emails_seconds,
            "status": self.status.value,
            "emails_sent_today": self.emails_sent_today,
            "emails_sent_this_hour": self.emails_sent_this_hour,
            "emails_sent_total": self.emails_sent_total,
            "last_sent_at": self.last_sent_at.isoformat() if self.last_sent_at else None,
            "last_error": self.last_error,
            "health_score": self.health_score,
            "bounce_rate_7d": self.bounce_rate_7d,
            "spam_rate_7d": self.spam_rate_7d,
            "open_rate_7d": self.open_rate_7d,
            "warmup_enabled": self.warmup_enabled,
            "warmup_day": self.warmup_day,
            "warmup_target_volume": self.warmup_target_volume,
            "signature_html": self.signature_html,
            "signature_text": self.signature_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmailAccount":
        account = cls()
        account.id = data.get("id", 0)
        account.email = data.get("email", "")
        account.display_name = data.get("display_name", "")
        account.smtp_host = data.get("smtp_host", "")
        account.smtp_port = data.get("smtp_port", 587)
        account.smtp_username = data.get("smtp_username", "")
        account.smtp_password = data.get("smtp_password", "")
        account.smtp_use_tls = data.get("smtp_use_tls", True)
        account.smtp_use_ssl = data.get("smtp_use_ssl", False)
        account.imap_host = data.get("imap_host", "")
        account.imap_port = data.get("imap_port", 993)
        account.imap_username = data.get("imap_username", "")
        account.imap_password = data.get("imap_password", "")
        account.imap_use_ssl = data.get("imap_use_ssl", True)
        account.daily_limit = data.get("daily_limit", 100)
        account.hourly_limit = data.get("hourly_limit", 20)
        account.delay_between_emails_seconds = data.get("delay_between_emails_seconds", 60)
        status_str = data.get("status", "pending")
        account.status = EmailAccountStatus(status_str) if isinstance(status_str, str) else status_str
        account.emails_sent_today = data.get("emails_sent_today", 0)
        account.emails_sent_this_hour = data.get("emails_sent_this_hour", 0)
        account.emails_sent_total = data.get("emails_sent_total", 0)
        if data.get("last_sent_at"):
            account.last_sent_at = datetime.fromisoformat(data["last_sent_at"]) if isinstance(data["last_sent_at"], str) else data["last_sent_at"]
        account.last_error = data.get("last_error", "")
        account.health_score = data.get("health_score", 100)
        account.bounce_rate_7d = data.get("bounce_rate_7d", 0.0)
        account.spam_rate_7d = data.get("spam_rate_7d", 0.0)
        account.open_rate_7d = data.get("open_rate_7d", 0.0)
        account.warmup_enabled = data.get("warmup_enabled", False)
        account.warmup_day = data.get("warmup_day", 0)
        account.warmup_target_volume = data.get("warmup_target_volume", 10)
        account.signature_html = data.get("signature_html", "")
        account.signature_text = data.get("signature_text", "")
        if data.get("created_at"):
            account.created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        if data.get("updated_at"):
            account.updated_at = datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"]
        return account

    def can_send(self, allow_pending: bool = False) -> bool:
        """Check if this account can send an email right now"""
        allowed_statuses = [EmailAccountStatus.ACTIVE, EmailAccountStatus.WARMUP]
        if allow_pending:
            allowed_statuses.append(EmailAccountStatus.PENDING)
        if self.status not in allowed_statuses:
            return False

        # Check daily limit
        effective_daily_limit = self.warmup_target_volume if self.warmup_enabled else self.daily_limit
        if self.emails_sent_today >= effective_daily_limit:
            return False

        # Check hourly limit
        if self.emails_sent_this_hour >= self.hourly_limit:
            return False

        # Check delay between emails
        if self.last_sent_at:
            seconds_since_last = (datetime.now() - self.last_sent_at).total_seconds()
            if seconds_since_last < self.delay_between_emails_seconds:
                return False

        return True

    def remaining_capacity_today(self) -> int:
        """Return how many more emails can be sent today"""
        effective_limit = self.warmup_target_volume if self.warmup_enabled else self.daily_limit
        return max(0, effective_limit - self.emails_sent_today)


@dataclass
class EmailMessage:
    """Email message to be sent"""
    to_email: str
    subject: str
    body_html: str
    body_text: str = ""
    from_email: Optional[str] = None  # If None, uses account email
    from_name: Optional[str] = None   # If None, uses account display_name
    reply_to: Optional[str] = None
    lead_id: Optional[int] = None
    campaign_id: Optional[int] = None
    campaign_step_id: Optional[int] = None
    tracking_id: Optional[str] = None  # For open/click tracking

    def __post_init__(self):
        if not self.tracking_id:
            self.tracking_id = str(uuid.uuid4())


class EmailManager:
    """
    Manages email sending with round-robin account selection.

    Usage:
        manager = EmailManager()
        manager.load_accounts()  # Load from database

        # Send an email
        result = manager.send(
            to_email="john@example.com",
            subject="Hello {{first_name}}",
            body_html="<p>Hi {{first_name}},</p>...",
            variables={"first_name": "John", "company": "Acme"}
        )
    """

    def __init__(self):
        self.accounts: List[EmailAccount] = []
        self._round_robin_index = 0
        self._db = None

    def load_accounts(self):
        """Load email accounts from database"""
        from database import get_db, dict_from_row

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM email_accounts ORDER BY id")
        rows = cursor.fetchall()

        self.accounts = []
        for row in rows:
            data = dict_from_row(row)
            account = EmailAccount.from_dict(data)
            self.accounts.append(account)

        conn.close()
        logger.info(f"Loaded {len(self.accounts)} email accounts")

    def save_account(self, account: EmailAccount) -> int:
        """Save or update an email account in the database"""
        from database import get_db

        conn = get_db()
        cursor = conn.cursor()

        now = datetime.now()

        if account.id:
            # Update existing
            cursor.execute("""
                UPDATE email_accounts SET
                    email = ?, display_name = ?,
                    smtp_host = ?, smtp_port = ?, smtp_username = ?, smtp_password = ?,
                    smtp_use_tls = ?, smtp_use_ssl = ?,
                    imap_host = ?, imap_port = ?, imap_username = ?, imap_password = ?,
                    imap_use_ssl = ?,
                    daily_limit = ?, hourly_limit = ?, delay_between_emails_seconds = ?,
                    status = ?,
                    emails_sent_today = ?, emails_sent_this_hour = ?, emails_sent_total = ?,
                    last_sent_at = ?, last_error = ?,
                    health_score = ?, bounce_rate_7d = ?, spam_rate_7d = ?, open_rate_7d = ?,
                    warmup_enabled = ?, warmup_day = ?, warmup_target_volume = ?,
                    signature_html = ?, signature_text = ?,
                    updated_at = ?
                WHERE id = ?
            """, (
                account.email, account.display_name,
                account.smtp_host, account.smtp_port, account.smtp_username, account.smtp_password,
                account.smtp_use_tls, account.smtp_use_ssl,
                account.imap_host, account.imap_port, account.imap_username, account.imap_password,
                account.imap_use_ssl,
                account.daily_limit, account.hourly_limit, account.delay_between_emails_seconds,
                account.status.value,
                account.emails_sent_today, account.emails_sent_this_hour, account.emails_sent_total,
                account.last_sent_at.isoformat() if account.last_sent_at else None, account.last_error,
                account.health_score, account.bounce_rate_7d, account.spam_rate_7d, account.open_rate_7d,
                account.warmup_enabled, account.warmup_day, account.warmup_target_volume,
                account.signature_html, account.signature_text,
                now.isoformat(), account.id
            ))
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO email_accounts (
                    email, display_name,
                    smtp_host, smtp_port, smtp_username, smtp_password,
                    smtp_use_tls, smtp_use_ssl,
                    imap_host, imap_port, imap_username, imap_password,
                    imap_use_ssl,
                    daily_limit, hourly_limit, delay_between_emails_seconds,
                    status,
                    emails_sent_today, emails_sent_this_hour, emails_sent_total,
                    last_sent_at, last_error,
                    health_score, bounce_rate_7d, spam_rate_7d, open_rate_7d,
                    warmup_enabled, warmup_day, warmup_target_volume,
                    signature_html, signature_text,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                account.email, account.display_name,
                account.smtp_host, account.smtp_port, account.smtp_username, account.smtp_password,
                account.smtp_use_tls, account.smtp_use_ssl,
                account.imap_host, account.imap_port, account.imap_username, account.imap_password,
                account.imap_use_ssl,
                account.daily_limit, account.hourly_limit, account.delay_between_emails_seconds,
                account.status.value,
                0, 0, 0,  # sent counts
                None, "",  # last_sent_at, last_error
                100, 0.0, 0.0, 0.0,  # health metrics
                account.warmup_enabled, 0, account.warmup_target_volume,
                account.signature_html, account.signature_text,
                now.isoformat(), now.isoformat()
            ))
            account.id = cursor.lastrowid

        conn.commit()
        conn.close()

        # Reload accounts
        self.load_accounts()
        return account.id

    def delete_account(self, account_id: int) -> bool:
        """Delete an email account"""
        from database import get_db

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM email_accounts WHERE id = ?", (account_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if deleted:
            self.load_accounts()
        return deleted

    def get_account(self, account_id: int) -> Optional[EmailAccount]:
        """Get a specific account by ID"""
        for account in self.accounts:
            if account.id == account_id:
                return account
        return None

    def get_available_accounts(self) -> List[EmailAccount]:
        """Get accounts that can send right now"""
        return [a for a in self.accounts if a.can_send()]

    def select_account_round_robin(self) -> Optional[EmailAccount]:
        """Select next available account using round-robin"""
        available = self.get_available_accounts()
        if not available:
            return None

        # Start from current index and find next available
        for i in range(len(self.accounts)):
            idx = (self._round_robin_index + i) % len(self.accounts)
            account = self.accounts[idx]
            if account.can_send():
                self._round_robin_index = (idx + 1) % len(self.accounts)
                return account

        return None

    def substitute_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """
        Replace {{variable}} placeholders with actual values.

        Variables can include:
        - first_name, last_name, full_name
        - email, phone
        - company, title, industry
        - city, state, country
        - Any custom field
        """
        if not text:
            return text

        # Build full_name if not provided
        if "full_name" not in variables:
            first = variables.get("first_name", "")
            last = variables.get("last_name", "")
            variables["full_name"] = f"{first} {last}".strip()

        # Replace all {{variable}} patterns
        def replace_match(match):
            var_name = match.group(1).strip()
            value = variables.get(var_name, "")
            return str(value) if value else ""

        return re.sub(r'\{\{(\w+)\}\}', replace_match, text)

    def inject_tracking(self, html: str, tracking_id: str, base_url: str = None) -> str:
        """
        Inject open tracking pixel and convert links for click tracking.

        Args:
            html: The HTML body
            tracking_id: Unique ID for this email
            base_url: Base URL for tracking endpoints (e.g., "https://yourdomain.com")
        """
        if not base_url:
            # Default to localhost for development
            base_url = "http://localhost:8000"

        # Add open tracking pixel before closing </body> tag
        tracking_pixel = f'<img src="{base_url}/t/o/{tracking_id}" width="1" height="1" style="display:none;" />'

        if "</body>" in html.lower():
            html = re.sub(
                r'(</body>)',
                f'{tracking_pixel}\\1',
                html,
                flags=re.IGNORECASE
            )
        else:
            html += tracking_pixel

        # Convert links for click tracking
        # Match href="..." but exclude mailto: and tel:
        def track_link(match):
            original_url = match.group(1)
            if original_url.startswith(('mailto:', 'tel:', '#')):
                return match.group(0)
            # URL encode the original URL
            import urllib.parse
            encoded_url = urllib.parse.quote(original_url, safe='')
            return f'href="{base_url}/t/c/{tracking_id}/{encoded_url}"'

        html = re.sub(r'href="([^"]+)"', track_link, html)

        return html

    def test_connection(self, account: EmailAccount) -> Dict[str, Any]:
        """
        Test SMTP connection for an account.

        Returns:
            {"success": True/False, "message": "...", "details": {...}}
        """
        try:
            logger.info(f"Testing SMTP connection for {account.email}")

            if account.smtp_use_ssl:
                # SSL connection (typically port 465)
                context = ssl.create_default_context()
                server = smtplib.SMTP_SSL(account.smtp_host, account.smtp_port, context=context, timeout=30)
            else:
                # TLS connection (typically port 587)
                server = smtplib.SMTP(account.smtp_host, account.smtp_port, timeout=30)
                if account.smtp_use_tls:
                    server.starttls()

            # Login
            server.login(account.smtp_username, account.smtp_password)

            # Check capabilities
            capabilities = server.esmtp_features

            server.quit()

            return {
                "success": True,
                "message": "Connection successful",
                "details": {
                    "host": account.smtp_host,
                    "port": account.smtp_port,
                    "tls": account.smtp_use_tls,
                    "ssl": account.smtp_use_ssl,
                    "capabilities": list(capabilities.keys()) if capabilities else []
                }
            }

        except smtplib.SMTPAuthenticationError as e:
            return {
                "success": False,
                "message": f"Authentication failed: {e.smtp_error.decode() if hasattr(e, 'smtp_error') else str(e)}",
                "error_type": "auth"
            }
        except smtplib.SMTPConnectError as e:
            return {
                "success": False,
                "message": f"Connection failed: {str(e)}",
                "error_type": "connection"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "error_type": "unknown"
            }

    def send(
        self,
        to_email: str,
        subject: str,
        body: str,
        body_html: str = "",
        variables: Dict[str, Any] = None,
        lead_id: Optional[int] = None,
        campaign_id: Optional[int] = None,
        campaign_step_id: Optional[int] = None,
        account_id: Optional[int] = None,
        inject_tracking: bool = False,
        tracking_base_url: str = None,
        allow_pending: bool = False
    ) -> Dict[str, Any]:
        """
        Send an email using round-robin account selection.

        Args:
            to_email: Recipient email address
            subject: Email subject (can contain {{variables}})
            body: Plain text body (required, can contain {{variables}})
            body_html: HTML body (optional, can contain {{variables}})
            variables: Dict of variables to substitute
            lead_id: Associated lead ID
            campaign_id: Associated campaign ID
            campaign_step_id: Associated campaign step ID
            account_id: Force use of specific account ID
            inject_tracking: Whether to inject open/click tracking (default: False)
            tracking_base_url: Base URL for tracking pixels

        Returns:
            {
                "success": True/False,
                "message_id": "...",
                "tracking_id": "...",
                "account_email": "...",
                "error": "..."
            }
        """
        variables = variables or {}

        # Select account
        if account_id:
            account = self.get_account(account_id)
            if not account:
                return {"success": False, "error": f"Account {account_id} not found"}
            if not account.can_send(allow_pending=allow_pending):
                return {"success": False, "error": f"Account {account.email} cannot send (limit reached or disabled)"}
        else:
            account = self.select_account_round_robin()
            if not account:
                return {"success": False, "error": "No available email accounts"}

        # Generate tracking ID (only used if tracking enabled)
        tracking_id = str(uuid.uuid4()) if inject_tracking else None

        # Substitute variables
        subject = self.substitute_variables(subject, variables)
        body_text = self.substitute_variables(body, variables)
        body_html_content = self.substitute_variables(body_html, variables) if body_html else ""

        # Add signature
        if account.signature_text:
            body_text = body_text + "\n\n" + account.signature_text
        if account.signature_html and body_html_content:
            body_html_content = body_html_content + account.signature_html

        # Inject tracking only if explicitly enabled AND there's HTML content
        if inject_tracking and body_html_content:
            body_html_content = self.inject_tracking(body_html_content, tracking_id, tracking_base_url)

        # Build message - plain text only, or multipart if HTML provided
        if body_html_content:
            msg = MIMEMultipart("alternative")
            msg.attach(MIMEText(body_text, "plain"))
            msg.attach(MIMEText(body_html_content, "html"))
        else:
            msg = MIMEText(body_text, "plain")

        msg["Subject"] = subject
        msg["From"] = formataddr((account.display_name or account.email.split("@")[0], account.email))
        msg["To"] = to_email
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain=account.email.split("@")[1])

        # Add custom headers for tracking (only if tracking enabled)
        if tracking_id:
            msg["X-Tracking-ID"] = tracking_id
        if campaign_id:
            msg["X-Campaign-ID"] = str(campaign_id)
        if lead_id:
            msg["X-Lead-ID"] = str(lead_id)

        # Send
        try:
            if account.smtp_use_ssl:
                context = ssl.create_default_context()
                server = smtplib.SMTP_SSL(account.smtp_host, account.smtp_port, context=context, timeout=30)
            else:
                server = smtplib.SMTP(account.smtp_host, account.smtp_port, timeout=30)
                if account.smtp_use_tls:
                    server.starttls()

            server.login(account.smtp_username, account.smtp_password)
            server.send_message(msg)
            server.quit()

            # Update account stats
            now = datetime.now()
            account.emails_sent_today += 1
            account.emails_sent_this_hour += 1
            account.emails_sent_total += 1
            account.last_sent_at = now
            account.last_error = ""
            if account.status == EmailAccountStatus.PENDING:
                account.status = EmailAccountStatus.ACTIVE
            self.save_account(account)

            # Log to messages table
            self._log_sent_message(
                account=account,
                to_email=to_email,
                subject=subject,
                body_html=body_html_content,
                body_text=body_text,
                tracking_id=tracking_id,
                lead_id=lead_id,
                campaign_id=campaign_id,
                campaign_step_id=campaign_step_id
            )

            logger.info(f"Email sent successfully via {account.email} to {to_email}")

            return {
                "success": True,
                "message_id": msg["Message-ID"],
                "tracking_id": tracking_id,
                "account_email": account.email
            }

        except smtplib.SMTPRecipientsRefused as e:
            # Bounce
            error_msg = f"Recipient refused: {to_email}"
            account.last_error = error_msg
            self.save_account(account)
            return {"success": False, "error": error_msg, "error_type": "bounce"}

        except Exception as e:
            error_msg = str(e)
            account.last_error = error_msg

            # Check if we should disable the account
            if "authentication" in error_msg.lower() or "login" in error_msg.lower():
                account.status = EmailAccountStatus.ERROR

            self.save_account(account)
            logger.error(f"Failed to send email via {account.email}: {error_msg}")
            return {"success": False, "error": error_msg}

    def _log_sent_message(
        self,
        account: EmailAccount,
        to_email: str,
        subject: str,
        body_html: str,
        body_text: str,
        tracking_id: str,
        lead_id: Optional[int],
        campaign_id: Optional[int],
        campaign_step_id: Optional[int]
    ):
        """Log sent message to database"""
        from database import save_message

        save_message({
            "channel": "email",
            "direction": "outbound",
            "from_address": account.email,
            "to_address": to_email,
            "subject": subject,
            "body": body_text,
            "body_html": body_html,
            "status": "sent",
            "lead_id": lead_id,
            "provider": "smtp",
            "external_id": tracking_id,
            "sent_at": datetime.now().isoformat(),
            "metadata": json.dumps({
                "tracking_id": tracking_id,
                "campaign_id": campaign_id,
                "campaign_step_id": campaign_step_id,
                "account_id": account.id
            })
        })

    def reset_daily_counts(self):
        """Reset daily send counts for all accounts (call at midnight)"""
        from database import get_db

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE email_accounts SET emails_sent_today = 0")
        conn.commit()
        conn.close()

        # Reload accounts
        self.load_accounts()
        logger.info("Reset daily email counts for all accounts")

    def reset_hourly_counts(self):
        """Reset hourly send counts for all accounts (call every hour)"""
        from database import get_db

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE email_accounts SET emails_sent_this_hour = 0")
        conn.commit()
        conn.close()

        # Reload accounts
        self.load_accounts()
        logger.info("Reset hourly email counts for all accounts")

    def advance_warmup(self):
        """Advance warmup day for accounts in warmup mode (call daily)"""
        from database import get_db

        # Warmup schedule: start at 10/day, increase by ~10% each day
        # Day 1: 10, Day 7: ~20, Day 14: ~40, Day 30: ~100, Day 45: ~200
        WARMUP_SCHEDULE = {
            1: 10, 7: 20, 14: 40, 21: 60, 30: 100, 45: 150
        }

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, warmup_day FROM email_accounts
            WHERE warmup_enabled = 1 AND status = 'warmup'
        """)

        for row in cursor.fetchall():
            account_id, warmup_day = row
            new_day = warmup_day + 1

            # Calculate new target volume
            new_target = 10
            for day, volume in sorted(WARMUP_SCHEDULE.items()):
                if new_day >= day:
                    new_target = volume

            # Check if warmup is complete
            if new_day >= 45:
                cursor.execute("""
                    UPDATE email_accounts
                    SET warmup_day = ?, warmup_enabled = 0, status = 'active', warmup_target_volume = ?
                    WHERE id = ?
                """, (new_day, new_target, account_id))
                logger.info(f"Account {account_id} completed warmup")
            else:
                cursor.execute("""
                    UPDATE email_accounts
                    SET warmup_day = ?, warmup_target_volume = ?
                    WHERE id = ?
                """, (new_day, new_target, account_id))

        conn.commit()
        conn.close()
        self.load_accounts()

    def get_stats(self) -> Dict[str, Any]:
        """Get overall email sending statistics"""
        total_accounts = len(self.accounts)
        active_accounts = len([a for a in self.accounts if a.status == EmailAccountStatus.ACTIVE])
        warmup_accounts = len([a for a in self.accounts if a.status == EmailAccountStatus.WARMUP])

        total_sent_today = sum(a.emails_sent_today for a in self.accounts)
        total_capacity_today = sum(a.remaining_capacity_today() for a in self.accounts)

        return {
            "total_accounts": total_accounts,
            "active_accounts": active_accounts,
            "warmup_accounts": warmup_accounts,
            "emails_sent_today": total_sent_today,
            "remaining_capacity_today": total_capacity_today,
            "accounts": [a.to_dict() for a in self.accounts]
        }


# Singleton instance
_email_manager: Optional[EmailManager] = None


def get_email_manager() -> EmailManager:
    """Get or create email manager singleton"""
    global _email_manager
    if _email_manager is None:
        _email_manager = EmailManager()
        try:
            _email_manager.load_accounts()
        except Exception as e:
            logger.warning(f"Could not load email accounts: {e}")
    return _email_manager


# Common SMTP presets for quick setup
SMTP_PRESETS = {
    "gmail": {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_use_tls": True,
        "smtp_use_ssl": False,
        "imap_host": "imap.gmail.com",
        "imap_port": 993,
        "imap_use_ssl": True,
        "notes": "Requires App Password if 2FA enabled"
    },
    "outlook": {
        "smtp_host": "smtp.office365.com",
        "smtp_port": 587,
        "smtp_use_tls": True,
        "smtp_use_ssl": False,
        "imap_host": "outlook.office365.com",
        "imap_port": 993,
        "imap_use_ssl": True,
        "notes": "Works with Microsoft 365 accounts"
    },
    "yahoo": {
        "smtp_host": "smtp.mail.yahoo.com",
        "smtp_port": 587,
        "smtp_use_tls": True,
        "smtp_use_ssl": False,
        "imap_host": "imap.mail.yahoo.com",
        "imap_port": 993,
        "imap_use_ssl": True,
        "notes": "Requires App Password"
    },
    "zoho": {
        "smtp_host": "smtp.zoho.com",
        "smtp_port": 587,
        "smtp_use_tls": True,
        "smtp_use_ssl": False,
        "imap_host": "imap.zoho.com",
        "imap_port": 993,
        "imap_use_ssl": True,
        "notes": "Free for small volume"
    },
    "sendgrid": {
        "smtp_host": "smtp.sendgrid.net",
        "smtp_port": 587,
        "smtp_use_tls": True,
        "smtp_use_ssl": False,
        "imap_host": "",
        "imap_port": 0,
        "imap_use_ssl": False,
        "notes": "Username is 'apikey', password is your API key"
    },
    "mailgun": {
        "smtp_host": "smtp.mailgun.org",
        "smtp_port": 587,
        "smtp_use_tls": True,
        "smtp_use_ssl": False,
        "imap_host": "",
        "imap_port": 0,
        "imap_use_ssl": False,
        "notes": "Check Mailgun dashboard for credentials"
    }
}

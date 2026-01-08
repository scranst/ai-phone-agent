"""
VersaBox CRM Database Layer

SQLite database for leads, interactions, campaigns, and sequences.
"""

import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).parent / "versabox.db"


def get_db() -> sqlite3.Connection:
    """Get database connection with row factory"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def dict_from_row(row: sqlite3.Row) -> dict:
    """Convert sqlite3.Row to dict"""
    return dict(row) if row else None


def normalize_phone(phone: str) -> str:
    """Normalize phone number to E.164 format (digits only)"""
    if not phone:
        return ""
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    # Add country code if missing (assume US)
    if len(digits) == 10:
        digits = "1" + digits
    return digits


def init_db():
    """Initialize database schema"""
    conn = get_db()
    cursor = conn.cursor()

    # Leads table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        -- Contact Information (email or phone required)
        email TEXT,
        phone TEXT,
        phone_normalized TEXT,
        first_name TEXT,
        last_name TEXT,
        linkedin_url TEXT,
        phone_type TEXT,

        -- Company/Professional Info
        company TEXT,
        title TEXT,
        industry TEXT,
        website TEXT,
        company_linkedin_url TEXT,
        company_size TEXT,
        employee_count INTEGER,
        revenue TEXT,
        funding_stage TEXT,
        seniority TEXT,
        department TEXT,
        technologies TEXT,

        -- Location
        address TEXT,
        city TEXT,
        state TEXT,
        country TEXT,
        timezone TEXT,

        -- Lifecycle & Scoring
        status TEXT DEFAULT 'NEW',
        lead_score INTEGER DEFAULT 0,
        source TEXT,
        source_details TEXT,

        -- AI/Sentiment
        sentiment_status TEXT,
        sentiment_confidence REAL,
        sentiment_decay_at DATETIME,

        -- Engagement Tracking
        last_contacted_at DATETIME,
        last_replied_at DATETIME,
        email_verified INTEGER DEFAULT 0,
        phone_verified INTEGER DEFAULT 0,
        email_validation_result TEXT,
        phone_validation_result TEXT,

        -- Personalization
        icebreaker TEXT,
        trigger_event TEXT,
        trigger_event_date DATE,
        pain_points TEXT,
        custom_1 TEXT,
        custom_2 TEXT,
        custom_3 TEXT,
        custom_4 TEXT,
        custom_5 TEXT,
        custom_fields TEXT,
        notes TEXT,

        -- Compliance
        do_not_contact INTEGER DEFAULT 0,
        auto_pilot_disabled INTEGER DEFAULT 0,
        dnc_detected_at DATETIME,
        dnc_trigger_message_id INTEGER,

        -- Timestamps
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_phone ON leads(phone_normalized)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_email ON leads(email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_sentiment ON leads(sentiment_status)")

    # Interactions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id INTEGER NOT NULL,

        channel TEXT NOT NULL,
        direction TEXT NOT NULL,
        status TEXT,

        subject TEXT,
        body TEXT,

        duration_seconds INTEGER,
        recording_path TEXT,
        transcript TEXT,
        summary TEXT,
        objective TEXT,
        outcome TEXT,

        queued_at DATETIME,
        sent_at DATETIME,
        delivered_at DATETIME,
        opened_at DATETIME,
        clicked_at DATETIME,
        replied_at DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (lead_id) REFERENCES leads(id)
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_lead ON interactions(lead_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_channel ON interactions(channel)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at)")

    # Lead lists table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lead_lists (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        filter_criteria TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lead_list_members (
        lead_list_id INTEGER,
        lead_id INTEGER,
        added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (lead_list_id, lead_id),
        FOREIGN KEY (lead_list_id) REFERENCES lead_lists(id),
        FOREIGN KEY (lead_id) REFERENCES leads(id)
    )
    """)

    # Campaigns table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS campaigns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        status TEXT DEFAULT 'draft',

        timezone TEXT DEFAULT 'America/Los_Angeles',
        execution_days TEXT,
        execution_hours_start TEXT DEFAULT '09:00',
        execution_hours_end TEXT DEFAULT '17:00',
        max_actions_per_day INTEGER,

        exit_on_reply INTEGER DEFAULT 1,
        exit_on_booking INTEGER DEFAULT 1,
        exit_on_dnc INTEGER DEFAULT 1,

        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        started_at DATETIME,
        completed_at DATETIME
    )
    """)

    # Campaign steps table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS campaign_steps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER NOT NULL,
        step_number INTEGER NOT NULL,

        delay_days INTEGER DEFAULT 0,
        delay_hours INTEGER DEFAULT 0,
        delay_type TEXT DEFAULT 'from_enrollment',

        channel TEXT NOT NULL,

        subject TEXT,
        body TEXT,
        template_id INTEGER,
        rvm_audio_path TEXT,

        condition_type TEXT,
        condition_value TEXT,

        call_objective TEXT,
        call_max_duration INTEGER,

        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (campaign_id) REFERENCES campaigns(id),
        UNIQUE(campaign_id, step_number)
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_steps_campaign ON campaign_steps(campaign_id, step_number)")

    # Campaign enrollments table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS campaign_enrollments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER NOT NULL,
        lead_id INTEGER NOT NULL,

        status TEXT DEFAULT 'active',
        current_step INTEGER DEFAULT 0,
        next_action_at DATETIME,

        enrolled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        paused_at DATETIME,
        completed_at DATETIME,
        exit_reason TEXT,

        FOREIGN KEY (campaign_id) REFERENCES campaigns(id),
        FOREIGN KEY (lead_id) REFERENCES leads(id),
        UNIQUE(campaign_id, lead_id)
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_enrollments_next ON campaign_enrollments(next_action_at, status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_enrollments_campaign ON campaign_enrollments(campaign_id, status)")

    # Step executions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS step_executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        enrollment_id INTEGER NOT NULL,
        step_id INTEGER NOT NULL,

        status TEXT DEFAULT 'pending',
        scheduled_at DATETIME,
        executed_at DATETIME,

        result TEXT,
        result_details TEXT,
        interaction_id INTEGER,
        error_message TEXT,

        FOREIGN KEY (enrollment_id) REFERENCES campaign_enrollments(id),
        FOREIGN KEY (step_id) REFERENCES campaign_steps(id),
        FOREIGN KEY (interaction_id) REFERENCES interactions(id)
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_executions_enrollment ON step_executions(enrollment_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_executions_status ON step_executions(status, scheduled_at)")

    # AI logs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ai_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id INTEGER,
        interaction_id INTEGER,
        log_type TEXT,
        action TEXT,
        confidence REAL,
        reasoning TEXT,
        tokens_used INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (lead_id) REFERENCES leads(id),
        FOREIGN KEY (interaction_id) REFERENCES interactions(id)
    )
    """)

    # =========================================================================
    # UNIFIED MESSAGES TABLE - SMS, Email, and future channels
    # =========================================================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        -- Link to lead (nullable for unknown senders)
        lead_id INTEGER,

        -- Channel: 'sms', 'email', 'whatsapp', 'call', etc.
        channel TEXT NOT NULL,

        -- Direction: 'inbound' or 'outbound'
        direction TEXT NOT NULL,

        -- Addresses (phone for SMS, email for email)
        from_address TEXT NOT NULL,
        to_address TEXT NOT NULL,

        -- Content
        subject TEXT,              -- For email (null for SMS)
        body TEXT,                 -- Plain text body
        body_html TEXT,            -- HTML body (for email)

        -- Threading
        thread_id TEXT,            -- Groups related messages (conversation ID)
        in_reply_to TEXT,          -- Message ID this is replying to

        -- Status tracking
        status TEXT DEFAULT 'sent',  -- queued, sending, sent, delivered, read, failed, bounced
        error_message TEXT,

        -- External IDs (for email message-id, SMS delivery receipts, etc.)
        external_id TEXT,
        provider TEXT,             -- 'modem', 'smtp', 'sendgrid', etc.

        -- Attachments (JSON array)
        attachments TEXT,

        -- Channel-specific metadata (JSON)
        metadata TEXT,

        -- Timestamps
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        sent_at DATETIME,
        delivered_at DATETIME,
        read_at DATETIME,

        FOREIGN KEY (lead_id) REFERENCES leads(id)
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_lead ON messages(lead_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_from ON messages(from_address)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_to ON messages(to_address)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at)")

    # Conversations table - aggregates messages by contact
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        -- Link to lead (nullable for unknown contacts)
        lead_id INTEGER,

        -- Contact identifier (phone number or email - normalized)
        contact_address TEXT NOT NULL,
        contact_name TEXT,

        -- Channels used in this conversation (JSON array: ["sms", "email"])
        channels TEXT DEFAULT '[]',

        -- Last message info for preview
        last_message_at DATETIME,
        last_message_preview TEXT,
        last_message_direction TEXT,
        last_message_channel TEXT,

        -- Unread count
        unread_count INTEGER DEFAULT 0,

        -- Timestamps
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (lead_id) REFERENCES leads(id),
        UNIQUE(contact_address)
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_lead ON conversations(lead_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_contact ON conversations(contact_address)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_last ON conversations(last_message_at)")

    # =========================================================================
    # EMAIL ACCOUNTS TABLE - For SMTP round-robin sending
    # =========================================================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS email_accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        -- Account identity
        email TEXT NOT NULL UNIQUE,
        display_name TEXT,

        -- SMTP settings
        smtp_host TEXT NOT NULL,
        smtp_port INTEGER DEFAULT 587,
        smtp_username TEXT NOT NULL,
        smtp_password TEXT NOT NULL,
        smtp_use_tls INTEGER DEFAULT 1,
        smtp_use_ssl INTEGER DEFAULT 0,

        -- IMAP settings (for reply detection)
        imap_host TEXT,
        imap_port INTEGER DEFAULT 993,
        imap_username TEXT,
        imap_password TEXT,
        imap_use_ssl INTEGER DEFAULT 1,

        -- Limits
        daily_limit INTEGER DEFAULT 100,
        hourly_limit INTEGER DEFAULT 20,
        delay_between_emails_seconds INTEGER DEFAULT 60,

        -- Status: pending, active, warmup, error, disabled
        status TEXT DEFAULT 'pending',

        -- Send tracking
        emails_sent_today INTEGER DEFAULT 0,
        emails_sent_this_hour INTEGER DEFAULT 0,
        emails_sent_total INTEGER DEFAULT 0,
        last_sent_at DATETIME,
        last_error TEXT,

        -- Health metrics
        health_score INTEGER DEFAULT 100,
        bounce_rate_7d REAL DEFAULT 0.0,
        spam_rate_7d REAL DEFAULT 0.0,
        open_rate_7d REAL DEFAULT 0.0,

        -- Warmup
        warmup_enabled INTEGER DEFAULT 0,
        warmup_day INTEGER DEFAULT 0,
        warmup_target_volume INTEGER DEFAULT 10,

        -- Signature
        signature_html TEXT,
        signature_text TEXT,

        -- Auto-disable tracking
        auto_disabled_at DATETIME,
        auto_disabled_reason TEXT,

        -- Timestamps
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_accounts_email ON email_accounts(email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_accounts_status ON email_accounts(status)")

    # =========================================================================
    # SETTINGS TABLE - All configuration stored here (not in JSON files)
    # =========================================================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        category TEXT DEFAULT 'general',
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_settings_category ON settings(category)")

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


# =============================================================================
# Lead CRUD
# =============================================================================

def create_lead(data: dict) -> int:
    """Create a new lead and return its ID"""
    conn = get_db()
    cursor = conn.cursor()

    # Normalize phone number
    if 'phone' in data:
        data['phone_normalized'] = normalize_phone(data['phone'])

    # Handle JSON fields
    for field in ['technologies', 'pain_points', 'custom_fields']:
        if field in data and isinstance(data[field], (list, dict)):
            data[field] = json.dumps(data[field])

    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?' for _ in data])
    values = list(data.values())

    cursor.execute(f"INSERT INTO leads ({columns}) VALUES ({placeholders})", values)
    lead_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return lead_id


def get_lead(lead_id: int) -> Optional[dict]:
    """Get a lead by ID"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM leads WHERE id = ?", (lead_id,))
    row = cursor.fetchone()
    conn.close()
    return dict_from_row(row)


def get_lead_by_phone(phone: str) -> Optional[dict]:
    """Get a lead by phone number"""
    conn = get_db()
    cursor = conn.cursor()
    normalized = normalize_phone(phone)
    cursor.execute("SELECT * FROM leads WHERE phone_normalized = ?", (normalized,))
    row = cursor.fetchone()
    conn.close()
    return dict_from_row(row)


def get_lead_by_email(email: str) -> Optional[dict]:
    """Get a lead by email"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM leads WHERE email = ?", (email.lower(),))
    row = cursor.fetchone()
    conn.close()
    return dict_from_row(row)


def update_lead(lead_id: int, data: dict) -> bool:
    """Update a lead"""
    conn = get_db()
    cursor = conn.cursor()

    # Update timestamp
    data['updated_at'] = datetime.now().isoformat()

    # Normalize phone if provided
    if 'phone' in data:
        data['phone_normalized'] = normalize_phone(data['phone'])

    # Handle JSON fields
    for field in ['technologies', 'pain_points', 'custom_fields']:
        if field in data and isinstance(data[field], (list, dict)):
            data[field] = json.dumps(data[field])

    set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
    values = list(data.values()) + [lead_id]

    cursor.execute(f"UPDATE leads SET {set_clause} WHERE id = ?", values)
    success = cursor.rowcount > 0

    conn.commit()
    conn.close()
    return success


def delete_lead(lead_id: int) -> bool:
    """Delete a lead and its interactions"""
    conn = get_db()
    cursor = conn.cursor()

    # Delete interactions first
    cursor.execute("DELETE FROM interactions WHERE lead_id = ?", (lead_id,))
    # Delete list memberships
    cursor.execute("DELETE FROM lead_list_members WHERE lead_id = ?", (lead_id,))
    # Delete enrollments
    cursor.execute("DELETE FROM campaign_enrollments WHERE lead_id = ?", (lead_id,))
    # Delete lead
    cursor.execute("DELETE FROM leads WHERE id = ?", (lead_id,))
    success = cursor.rowcount > 0

    conn.commit()
    conn.close()
    return success


def delete_leads_bulk(lead_ids: List[int]) -> int:
    """Delete multiple leads and their related data. Returns count of deleted leads."""
    if not lead_ids:
        return 0

    conn = get_db()
    cursor = conn.cursor()

    placeholders = ','.join('?' * len(lead_ids))

    # Delete interactions first
    cursor.execute(f"DELETE FROM interactions WHERE lead_id IN ({placeholders})", lead_ids)
    # Delete list memberships
    cursor.execute(f"DELETE FROM lead_list_members WHERE lead_id IN ({placeholders})", lead_ids)
    # Delete enrollments
    cursor.execute(f"DELETE FROM campaign_enrollments WHERE lead_id IN ({placeholders})", lead_ids)
    # Delete leads
    cursor.execute(f"DELETE FROM leads WHERE id IN ({placeholders})", lead_ids)
    deleted_count = cursor.rowcount

    conn.commit()
    conn.close()
    return deleted_count


def search_leads(
    filters: dict = None,
    search: str = None,
    offset: int = 0,
    limit: int = 50,
    order_by: str = "created_at DESC"
) -> tuple[List[dict], int]:
    """
    Search leads with filters and pagination.
    Returns (leads, total_count)
    """
    conn = get_db()
    cursor = conn.cursor()

    where_clauses = []
    params = []

    if filters:
        if filters.get('status'):
            where_clauses.append("status = ?")
            params.append(filters['status'])
        if filters.get('sentiment_status'):
            where_clauses.append("sentiment_status = ?")
            params.append(filters['sentiment_status'])
        if filters.get('source'):
            where_clauses.append("source = ?")
            params.append(filters['source'])
        if filters.get('do_not_contact'):
            where_clauses.append("do_not_contact = 1")

    if search:
        search_term = f"%{search}%"
        # Also search combined first+last name for multi-word queries like "Neo Geo"
        where_clauses.append("""
            (first_name LIKE ? OR last_name LIKE ? OR email LIKE ?
             OR phone LIKE ? OR company LIKE ?
             OR (first_name || ' ' || last_name) LIKE ?)
        """)
        params.extend([search_term] * 6)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    # Get total count
    cursor.execute(f"SELECT COUNT(*) FROM leads WHERE {where_sql}", params)
    total = cursor.fetchone()[0]

    # Get paginated results
    cursor.execute(
        f"SELECT * FROM leads WHERE {where_sql} ORDER BY {order_by} LIMIT ? OFFSET ?",
        params + [limit, offset]
    )
    leads = [dict_from_row(row) for row in cursor.fetchall()]

    conn.close()
    return leads, total


def get_all_leads() -> List[dict]:
    """Get all leads"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM leads ORDER BY created_at DESC")
    leads = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    return leads


# =============================================================================
# Interaction CRUD
# =============================================================================

def log_interaction(lead_id: int, data: dict) -> int:
    """Log an interaction for a lead"""
    conn = get_db()
    cursor = conn.cursor()

    data['lead_id'] = lead_id

    # Handle transcript JSON
    if 'transcript' in data and isinstance(data['transcript'], list):
        data['transcript'] = json.dumps(data['transcript'])

    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?' for _ in data])
    values = list(data.values())

    cursor.execute(f"INSERT INTO interactions ({columns}) VALUES ({placeholders})", values)
    interaction_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return interaction_id


def get_lead_interactions(lead_id: int, limit: int = 50) -> List[dict]:
    """Get interactions for a lead"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM interactions WHERE lead_id = ? ORDER BY created_at DESC LIMIT ?",
        (lead_id, limit)
    )
    interactions = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    return interactions


# =============================================================================
# Lead Lists CRUD
# =============================================================================

def create_lead_list(name: str, description: str = "") -> int:
    """Create a new lead list"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO lead_lists (name, description) VALUES (?, ?)",
        (name, description)
    )
    list_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return list_id


def get_lead_lists() -> List[dict]:
    """Get all lead lists with counts"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ll.*, COUNT(llm.lead_id) as lead_count
        FROM lead_lists ll
        LEFT JOIN lead_list_members llm ON ll.id = llm.lead_list_id
        GROUP BY ll.id
        ORDER BY ll.created_at DESC
    """)
    lists = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    return lists


def add_leads_to_list(list_id: int, lead_ids: List[int]):
    """Add leads to a list"""
    conn = get_db()
    cursor = conn.cursor()
    for lead_id in lead_ids:
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO lead_list_members (lead_list_id, lead_id) VALUES (?, ?)",
                (list_id, lead_id)
            )
        except sqlite3.IntegrityError:
            pass  # Lead already in list
    conn.commit()
    conn.close()


def get_list_leads(list_id: int) -> List[dict]:
    """Get leads in a list"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT l.* FROM leads l
        JOIN lead_list_members llm ON l.id = llm.lead_id
        WHERE llm.lead_list_id = ?
        ORDER BY llm.added_at DESC
    """, (list_id,))
    leads = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    return leads


def delete_lead_list(list_id: int) -> bool:
    """Delete a lead list"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM lead_list_members WHERE lead_list_id = ?", (list_id,))
    cursor.execute("DELETE FROM lead_lists WHERE id = ?", (list_id,))
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return success


# =============================================================================
# Campaign CRUD
# =============================================================================

def create_campaign(data: dict) -> int:
    """Create a new campaign"""
    conn = get_db()
    cursor = conn.cursor()

    # Handle JSON fields
    if 'execution_days' in data and isinstance(data['execution_days'], list):
        data['execution_days'] = json.dumps(data['execution_days'])

    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?' for _ in data])
    values = list(data.values())

    cursor.execute(f"INSERT INTO campaigns ({columns}) VALUES ({placeholders})", values)
    campaign_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return campaign_id


def get_campaign(campaign_id: int) -> Optional[dict]:
    """Get a campaign by ID"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,))
    row = cursor.fetchone()
    conn.close()
    return dict_from_row(row)


def get_campaigns() -> List[dict]:
    """Get all campaigns with stats"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.*,
            (SELECT COUNT(*) FROM campaign_enrollments WHERE campaign_id = c.id) as total_enrolled,
            (SELECT COUNT(*) FROM campaign_enrollments WHERE campaign_id = c.id AND status = 'active') as active_count,
            (SELECT COUNT(*) FROM campaign_enrollments WHERE campaign_id = c.id AND status = 'completed') as completed_count
        FROM campaigns c
        ORDER BY c.created_at DESC
    """)
    campaigns = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    return campaigns


def update_campaign(campaign_id: int, data: dict) -> bool:
    """Update a campaign"""
    conn = get_db()
    cursor = conn.cursor()

    if 'execution_days' in data and isinstance(data['execution_days'], list):
        data['execution_days'] = json.dumps(data['execution_days'])

    set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
    values = list(data.values()) + [campaign_id]

    cursor.execute(f"UPDATE campaigns SET {set_clause} WHERE id = ?", values)
    success = cursor.rowcount > 0

    conn.commit()
    conn.close()
    return success


# =============================================================================
# Campaign Steps CRUD
# =============================================================================

def add_campaign_step(campaign_id: int, data: dict) -> int:
    """Add a step to a campaign"""
    conn = get_db()
    cursor = conn.cursor()

    data['campaign_id'] = campaign_id

    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?' for _ in data])
    values = list(data.values())

    cursor.execute(f"INSERT INTO campaign_steps ({columns}) VALUES ({placeholders})", values)
    step_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return step_id


def get_campaign_steps(campaign_id: int) -> List[dict]:
    """Get all steps for a campaign"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM campaign_steps WHERE campaign_id = ? ORDER BY step_number",
        (campaign_id,)
    )
    steps = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    return steps


def get_campaign_step(campaign_id: int, step_number: int) -> Optional[dict]:
    """Get a specific step"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM campaign_steps WHERE campaign_id = ? AND step_number = ?",
        (campaign_id, step_number)
    )
    row = cursor.fetchone()
    conn.close()
    return dict_from_row(row)


# =============================================================================
# Campaign Enrollments
# =============================================================================

def enroll_leads(campaign_id: int, lead_ids: List[int], next_action_at: datetime = None):
    """Enroll leads in a campaign"""
    conn = get_db()
    cursor = conn.cursor()

    if next_action_at is None:
        next_action_at = datetime.now()

    for lead_id in lead_ids:
        try:
            cursor.execute("""
                INSERT INTO campaign_enrollments (campaign_id, lead_id, next_action_at)
                VALUES (?, ?, ?)
            """, (campaign_id, lead_id, next_action_at.isoformat()))
        except sqlite3.IntegrityError:
            pass  # Already enrolled

    conn.commit()
    conn.close()


def get_due_enrollments(limit: int = 100) -> List[dict]:
    """Get enrollments that are due for action"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ce.*, l.*, c.name as campaign_name
        FROM campaign_enrollments ce
        JOIN leads l ON ce.lead_id = l.id
        JOIN campaigns c ON ce.campaign_id = c.id
        WHERE ce.status = 'active'
          AND ce.next_action_at <= datetime('now')
          AND c.status = 'active'
        ORDER BY ce.next_action_at
        LIMIT ?
    """, (limit,))
    enrollments = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    return enrollments


def update_enrollment(enrollment_id: int, data: dict) -> bool:
    """Update an enrollment"""
    conn = get_db()
    cursor = conn.cursor()

    set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
    values = list(data.values()) + [enrollment_id]

    cursor.execute(f"UPDATE campaign_enrollments SET {set_clause} WHERE id = ?", values)
    success = cursor.rowcount > 0

    conn.commit()
    conn.close()
    return success


def get_campaign_enrollments(campaign_id: int) -> List[dict]:
    """Get all enrollments for a campaign with lead info"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ce.*, l.first_name, l.last_name, l.email, l.phone, l.company
        FROM campaign_enrollments ce
        JOIN leads l ON ce.lead_id = l.id
        WHERE ce.campaign_id = ?
        ORDER BY ce.enrolled_at DESC
    """, (campaign_id,))
    enrollments = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    return enrollments


# =============================================================================
# CSV Import
# =============================================================================

def import_csv(filepath: str, field_mapping: dict) -> dict:
    """
    Import leads from CSV with field mapping.

    Args:
        filepath: Path to CSV file
        field_mapping: Dict mapping CSV columns to database fields
            e.g. {"Email": "email", "Phone Number": "phone"}

    Returns:
        {"imported": 50, "skipped": 3, "errors": [...]}
    """
    import csv

    result = {"imported": 0, "skipped": 0, "updated": 0, "errors": []}

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):  # Start at 2 (after header)
            try:
                # Map CSV columns to database fields
                lead_data = {}
                for csv_col, db_field in field_mapping.items():
                    if csv_col in row and row[csv_col]:
                        lead_data[db_field] = row[csv_col].strip()

                # Must have phone
                if 'phone' not in lead_data or not lead_data['phone']:
                    result['skipped'] += 1
                    continue

                # Check if lead exists
                existing = get_lead_by_phone(lead_data['phone'])
                if existing:
                    # Update existing lead
                    update_lead(existing['id'], lead_data)
                    result['updated'] += 1
                else:
                    # Create new lead
                    lead_data['source'] = 'csv_import'
                    create_lead(lead_data)
                    result['imported'] += 1

            except Exception as e:
                result['errors'].append(f"Row {row_num}: {str(e)}")

    return result


# =============================================================================
# Stats & Analytics
# =============================================================================

def get_lead_stats() -> dict:
    """Get lead statistics"""
    conn = get_db()
    cursor = conn.cursor()

    stats = {}

    # Total leads
    cursor.execute("SELECT COUNT(*) FROM leads")
    stats['total'] = cursor.fetchone()[0]

    # By status
    cursor.execute("""
        SELECT status, COUNT(*) as count
        FROM leads
        GROUP BY status
    """)
    stats['by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}

    # By sentiment
    cursor.execute("""
        SELECT sentiment_status, COUNT(*) as count
        FROM leads
        WHERE sentiment_status IS NOT NULL
        GROUP BY sentiment_status
    """)
    stats['by_sentiment'] = {row['sentiment_status']: row['count'] for row in cursor.fetchall()}

    # Recent (last 7 days)
    cursor.execute("""
        SELECT COUNT(*) FROM leads
        WHERE created_at >= datetime('now', '-7 days')
    """)
    stats['recent_7d'] = cursor.fetchone()[0]

    conn.close()
    return stats


# =============================================================================
# Messages CRUD (Unified Inbox)
# =============================================================================

def normalize_address(address: str, channel: str = 'sms') -> str:
    """Normalize an address (phone or email) for consistent lookups"""
    if not address:
        return ""
    if channel == 'email':
        return address.lower().strip()
    else:
        # Phone number - extract digits
        return normalize_phone(address)


def save_message(data: dict) -> int:
    """
    Save a message to the unified messages table.

    Required fields:
    - channel: 'sms', 'email', etc.
    - direction: 'inbound' or 'outbound'
    - from_address, to_address
    - body

    Returns: message ID
    """
    conn = get_db()
    cursor = conn.cursor()

    # Determine contact address (the other party)
    if data['direction'] == 'inbound':
        contact_address = normalize_address(data['from_address'], data['channel'])
    else:
        contact_address = normalize_address(data['to_address'], data['channel'])

    # Try to find or create lead by address
    if not data.get('lead_id') and contact_address:
        if data['channel'] == 'email':
            lead = get_lead_by_email(contact_address)
        else:
            lead = get_lead_by_phone(contact_address)
        if lead:
            data['lead_id'] = lead['id']

    # Set thread_id if not provided (group by contact address)
    if not data.get('thread_id'):
        data['thread_id'] = contact_address

    # Handle JSON fields
    for field in ['attachments', 'metadata']:
        if field in data and isinstance(data[field], (list, dict)):
            data[field] = json.dumps(data[field])

    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?' for _ in data])
    values = list(data.values())

    cursor.execute(f"INSERT INTO messages ({columns}) VALUES ({placeholders})", values)
    message_id = cursor.lastrowid

    # Update or create conversation
    _update_conversation(cursor, contact_address, data)

    conn.commit()
    conn.close()
    return message_id


def _update_conversation(cursor, contact_address: str, message_data: dict):
    """Update or create conversation entry after saving a message"""
    if not contact_address:
        return

    # Check if conversation exists
    cursor.execute(
        "SELECT id, channels FROM conversations WHERE contact_address = ?",
        (contact_address,)
    )
    row = cursor.fetchone()

    preview = (message_data.get('body') or '')[:100]
    channel = message_data.get('channel', 'sms')
    now = datetime.now().isoformat()

    if row:
        # Update existing conversation
        conv_id, channels_json = row
        channels = json.loads(channels_json) if channels_json else []
        if channel not in channels:
            channels.append(channel)

        # Update unread count for inbound messages
        unread_increment = 1 if message_data.get('direction') == 'inbound' else 0

        cursor.execute("""
            UPDATE conversations SET
                last_message_at = ?,
                last_message_preview = ?,
                last_message_direction = ?,
                last_message_channel = ?,
                channels = ?,
                unread_count = unread_count + ?,
                updated_at = ?
            WHERE id = ?
        """, (
            now, preview, message_data.get('direction'), channel,
            json.dumps(channels), unread_increment, now, conv_id
        ))
    else:
        # Create new conversation
        cursor.execute("""
            INSERT INTO conversations (
                lead_id, contact_address, contact_name, channels,
                last_message_at, last_message_preview, last_message_direction,
                last_message_channel, unread_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message_data.get('lead_id'),
            contact_address,
            None,  # contact_name - will be filled from lead if available
            json.dumps([channel]),
            now, preview, message_data.get('direction'), channel,
            1 if message_data.get('direction') == 'inbound' else 0
        ))


def get_conversations(limit: int = 50, offset: int = 0) -> List[dict]:
    """Get conversations list for inbox, ordered by most recent"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT c.*,
            l.first_name, l.last_name, l.company, l.email, l.phone
        FROM conversations c
        LEFT JOIN leads l ON c.lead_id = l.id
        ORDER BY c.last_message_at DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))

    conversations = []
    for row in cursor.fetchall():
        conv = dict_from_row(row)
        # Build display name
        if conv.get('first_name') or conv.get('last_name'):
            conv['display_name'] = f"{conv.get('first_name', '')} {conv.get('last_name', '')}".strip()
        elif conv.get('company'):
            conv['display_name'] = conv['company']
        else:
            conv['display_name'] = conv['contact_address']
        # Parse channels JSON
        conv['channels'] = json.loads(conv.get('channels') or '[]')
        conversations.append(conv)

    conn.close()
    return conversations


def get_conversation_messages(contact_address: str, limit: int = 100, offset: int = 0) -> List[dict]:
    """Get messages for a specific conversation"""
    conn = get_db()
    cursor = conn.cursor()

    normalized = normalize_address(contact_address)

    cursor.execute("""
        SELECT * FROM messages
        WHERE thread_id = ? OR from_address = ? OR to_address = ?
        ORDER BY created_at ASC
        LIMIT ? OFFSET ?
    """, (normalized, normalized, normalized, limit, offset))

    messages = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    return messages


def mark_conversation_read(contact_address: str):
    """Mark all messages in a conversation as read"""
    conn = get_db()
    cursor = conn.cursor()

    normalized = normalize_address(contact_address)

    # Mark messages as read
    cursor.execute("""
        UPDATE messages SET read_at = datetime('now')
        WHERE (from_address = ? OR thread_id = ?)
        AND direction = 'inbound' AND read_at IS NULL
    """, (normalized, normalized))

    # Reset unread count
    cursor.execute("""
        UPDATE conversations SET unread_count = 0
        WHERE contact_address = ?
    """, (normalized,))

    conn.commit()
    conn.close()


def get_unread_count() -> int:
    """Get total unread message count across all conversations"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(SUM(unread_count), 0) FROM conversations")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def is_thread_autopilot_disabled(contact_address: str) -> bool:
    """Check if autopilot is disabled for a specific conversation thread"""
    conn = get_db()
    cursor = conn.cursor()

    normalized = normalize_address(contact_address)

    cursor.execute(
        "SELECT autopilot_disabled FROM conversations WHERE contact_address = ?",
        (normalized,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return bool(row[0])
    return False


def set_thread_autopilot(contact_address: str, enabled: bool):
    """Set autopilot enabled/disabled for a specific conversation thread"""
    conn = get_db()
    cursor = conn.cursor()

    normalized = normalize_address(contact_address)
    disabled = 0 if enabled else 1

    # Update if exists, create if not
    cursor.execute(
        "SELECT id FROM conversations WHERE contact_address = ?",
        (normalized,)
    )
    row = cursor.fetchone()

    if row:
        cursor.execute(
            "UPDATE conversations SET autopilot_disabled = ? WHERE contact_address = ?",
            (disabled, normalized)
        )
    else:
        # Create conversation if it doesn't exist
        cursor.execute("""
            INSERT INTO conversations (contact_address, autopilot_disabled, channels)
            VALUES (?, ?, '[]')
        """, (normalized, disabled))

    conn.commit()
    conn.close()


# =============================================================================
# Settings CRUD (replaces settings.json)
# =============================================================================

def get_setting(key: str, default: str = None) -> Optional[str]:
    """Get a single setting by key"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row['value'] if row else default


def set_setting(key: str, value: str, category: str = 'general'):
    """Set a single setting"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO settings (key, value, category, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            category = excluded.category,
            updated_at = datetime('now')
    """, (key, value, category))
    conn.commit()
    conn.close()


def get_settings_by_category(category: str) -> Dict[str, str]:
    """Get all settings in a category as a dict"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM settings WHERE category = ?", (category,))
    result = {row['key']: row['value'] for row in cursor.fetchall()}
    conn.close()
    return result


def get_all_settings() -> dict:
    """
    Get all settings as a nested dict matching the old settings.json structure.
    Reconstructs: {api_keys: {...}, agents: {...}, integrations: {...}, ...}
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value, category FROM settings")
    rows = cursor.fetchall()
    conn.close()

    # Build nested structure
    result = {
        'api_keys': {},
        'agents': {},
        'integrations': {},
    }

    for row in rows:
        key, value, category = row['key'], row['value'], row['category']

        # Try to parse JSON values (for agents, etc.)
        try:
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed_value = value

        if category == 'api_keys':
            result['api_keys'][key] = parsed_value
        elif category == 'agents':
            # Agent configs stored as agent_<id> with JSON value
            if key.startswith('agent_'):
                agent_id = key[6:]  # Remove 'agent_' prefix
                result['agents'][agent_id] = parsed_value
            else:
                result['agents'][key] = parsed_value
        elif category == 'integrations':
            result['integrations'][key] = parsed_value
        else:
            # Top-level settings (user info, etc.)
            result[key] = parsed_value

    return result


def set_settings_bulk(settings: dict, category_map: dict = None):
    """
    Save settings from a nested dict (like old settings.json structure).
    Flattens and stores each key with appropriate category.
    """
    if category_map is None:
        category_map = {
            'ANTHROPIC_API_KEY': 'api_keys',
            'OPENAI_API_KEY': 'api_keys',
            'GOOGLE_API_KEY': 'api_keys',
            'GOOGLE_CSE_ID': 'api_keys',
            'APIFY_API_KEY': 'api_keys',
            'AMADEUS_API_KEY': 'api_keys',
            'AMADEUS_API_SECRET': 'api_keys',
            'NEVERBOUNCE_API_KEY': 'api_keys',
            'PHONEVALIDATOR_API_KEY': 'api_keys',
            'CALENDAR_PROVIDER': 'integrations',
            'CAL_COM_API_KEY': 'integrations',
            'CAL_COM_EVENT_TYPE_ID': 'integrations',
            'CALENDLY_API_KEY': 'integrations',
            'CALENDLY_USER_URI': 'integrations',
        }

    conn = get_db()
    cursor = conn.cursor()

    def save_item(key: str, value, category: str):
        # Serialize complex values as JSON
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        else:
            value = str(value) if value is not None else ''

        cursor.execute("""
            INSERT INTO settings (key, value, category, updated_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                category = excluded.category,
                updated_at = datetime('now')
        """, (key, value, category))

    # Handle nested structures
    for key, value in settings.items():
        if key == 'api_keys' and isinstance(value, dict):
            for k, v in value.items():
                save_item(k, v, 'api_keys')
        elif key == 'agents' and isinstance(value, dict):
            for agent_id, agent_config in value.items():
                save_item(f'agent_{agent_id}', agent_config, 'agents')
        elif key == 'integrations' and isinstance(value, dict):
            for k, v in value.items():
                save_item(k, v, 'integrations')
        else:
            # Top-level setting - determine category
            category = category_map.get(key, 'user_info')
            save_item(key, value, category)

    conn.commit()
    conn.close()


def import_settings_from_json(json_path: str) -> bool:
    """Import settings from a JSON file (for migration)"""
    try:
        with open(json_path, 'r') as f:
            settings = json.load(f)
        set_settings_bulk(settings)
        return True
    except Exception as e:
        print(f"Error importing settings from {json_path}: {e}")
        return False


def migrate_db():
    """Run any necessary migrations"""
    conn = get_db()
    cursor = conn.cursor()

    # Check if email_accounts table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='email_accounts'")
    if not cursor.fetchone():
        print("Creating email_accounts table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS email_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            display_name TEXT,
            smtp_host TEXT NOT NULL,
            smtp_port INTEGER DEFAULT 587,
            smtp_username TEXT NOT NULL,
            smtp_password TEXT NOT NULL,
            smtp_use_tls INTEGER DEFAULT 1,
            smtp_use_ssl INTEGER DEFAULT 0,
            imap_host TEXT,
            imap_port INTEGER DEFAULT 993,
            imap_username TEXT,
            imap_password TEXT,
            imap_use_ssl INTEGER DEFAULT 1,
            daily_limit INTEGER DEFAULT 100,
            hourly_limit INTEGER DEFAULT 20,
            delay_between_emails_seconds INTEGER DEFAULT 60,
            status TEXT DEFAULT 'pending',
            emails_sent_today INTEGER DEFAULT 0,
            emails_sent_this_hour INTEGER DEFAULT 0,
            emails_sent_total INTEGER DEFAULT 0,
            last_sent_at DATETIME,
            last_error TEXT,
            health_score INTEGER DEFAULT 100,
            bounce_rate_7d REAL DEFAULT 0.0,
            spam_rate_7d REAL DEFAULT 0.0,
            open_rate_7d REAL DEFAULT 0.0,
            warmup_enabled INTEGER DEFAULT 0,
            warmup_day INTEGER DEFAULT 0,
            warmup_target_volume INTEGER DEFAULT 10,
            signature_html TEXT,
            signature_text TEXT,
            auto_disabled_at DATETIME,
            auto_disabled_reason TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_accounts_email ON email_accounts(email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_accounts_status ON email_accounts(status)")
        conn.commit()
        print("email_accounts table created!")

    # Check if autopilot_disabled column exists in conversations table
    cursor.execute("PRAGMA table_info(conversations)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'autopilot_disabled' not in columns:
        print("Adding autopilot_disabled column to conversations table...")
        try:
            cursor.execute("ALTER TABLE conversations ADD COLUMN autopilot_disabled INTEGER DEFAULT 0")
            conn.commit()
            print("autopilot_disabled column added!")
        except Exception as e:
            print(f"Error adding autopilot_disabled column: {e}")

    # Check if messages table exists (new unified inbox tables)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
    if not cursor.fetchone():
        print("Creating messages and conversations tables...")
        # Create messages table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id INTEGER,
            channel TEXT NOT NULL,
            direction TEXT NOT NULL,
            from_address TEXT NOT NULL,
            to_address TEXT NOT NULL,
            subject TEXT,
            body TEXT,
            body_html TEXT,
            thread_id TEXT,
            in_reply_to TEXT,
            status TEXT DEFAULT 'sent',
            error_message TEXT,
            external_id TEXT,
            provider TEXT,
            attachments TEXT,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            sent_at DATETIME,
            delivered_at DATETIME,
            read_at DATETIME,
            FOREIGN KEY (lead_id) REFERENCES leads(id)
        )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_lead ON messages(lead_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_from ON messages(from_address)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_to ON messages(to_address)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at)")

        # Create conversations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id INTEGER,
            contact_address TEXT NOT NULL,
            contact_name TEXT,
            channels TEXT DEFAULT '[]',
            last_message_at DATETIME,
            last_message_preview TEXT,
            last_message_direction TEXT,
            last_message_channel TEXT,
            unread_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (lead_id) REFERENCES leads(id),
            UNIQUE(contact_address)
        )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_lead ON conversations(lead_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_contact ON conversations(contact_address)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_last ON conversations(last_message_at)")
        conn.commit()
        print("Messages and conversations tables created!")

    # Check if phone column has NOT NULL constraint
    # SQLite doesn't allow modifying constraints directly, so we recreate the table
    cursor.execute("PRAGMA table_info(leads)")
    columns = cursor.fetchall()

    for col in columns:
        # col format: (cid, name, type, notnull, dflt_value, pk)
        if col[1] == 'phone' and col[3] == 1:  # notnull = 1
            print("Migrating leads table: removing NOT NULL from phone column...")

            try:
                # Disable foreign key checks during migration
                cursor.execute("PRAGMA foreign_keys = OFF")

                # Create new table without NOT NULL on phone
                cursor.execute("ALTER TABLE leads RENAME TO leads_old")

                # Recreate with updated schema (phone can be NULL)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS leads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT,
                    phone TEXT,
                    phone_normalized TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    linkedin_url TEXT,
                    phone_type TEXT,
                    company TEXT,
                    title TEXT,
                    industry TEXT,
                    website TEXT,
                    company_linkedin_url TEXT,
                    company_size TEXT,
                    employee_count INTEGER,
                    revenue TEXT,
                    funding_stage TEXT,
                    seniority TEXT,
                    department TEXT,
                    technologies TEXT,
                    address TEXT,
                    city TEXT,
                    state TEXT,
                    country TEXT,
                    timezone TEXT,
                    status TEXT DEFAULT 'NEW',
                    lead_score INTEGER DEFAULT 0,
                    source TEXT,
                    source_details TEXT,
                    sentiment_status TEXT,
                    sentiment_confidence REAL,
                    sentiment_decay_at DATETIME,
                    last_contacted_at DATETIME,
                    last_replied_at DATETIME,
                    email_verified INTEGER DEFAULT 0,
                    phone_verified INTEGER DEFAULT 0,
                    email_validation_result TEXT,
                    phone_validation_result TEXT,
                    icebreaker TEXT,
                    trigger_event TEXT,
                    trigger_event_date DATE,
                    pain_points TEXT,
                    custom_1 TEXT,
                    custom_2 TEXT,
                    custom_3 TEXT,
                    custom_4 TEXT,
                    custom_5 TEXT,
                    custom_fields TEXT,
                    notes TEXT,
                    do_not_contact INTEGER DEFAULT 0,
                    auto_pilot_disabled INTEGER DEFAULT 0,
                    dnc_detected_at DATETIME,
                    dnc_trigger_message_id INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)

                # Copy data
                cursor.execute("""
                    INSERT INTO leads SELECT * FROM leads_old
                """)

                # Drop old table
                cursor.execute("DROP TABLE leads_old")

                # Re-enable foreign key checks
                cursor.execute("PRAGMA foreign_keys = ON")

                conn.commit()
                print("Migration complete!")
            except Exception as e:
                print(f"Migration error: {e}")
                conn.rollback()
                # Re-enable foreign key checks even on error
                try:
                    cursor.execute("PRAGMA foreign_keys = ON")
                except:
                    pass
            break

    # Check if lead_lists table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lead_lists'")
    if not cursor.fetchone():
        print("Creating lead_lists and lead_list_members tables...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS lead_lists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS lead_list_members (
            lead_list_id INTEGER,
            lead_id INTEGER,
            added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (lead_list_id, lead_id),
            FOREIGN KEY (lead_list_id) REFERENCES lead_lists(id),
            FOREIGN KEY (lead_id) REFERENCES leads(id)
        )
        """)
        conn.commit()
        print("lead_lists tables created!")

    # Check if settings table exists and migrate from JSON if needed
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='settings'")
    if not cursor.fetchone():
        print("Creating settings table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_settings_category ON settings(category)")
        conn.commit()
        print("Settings table created!")

    # Check if settings table is empty and import from settings.json
    cursor.execute("SELECT COUNT(*) FROM settings")
    settings_count = cursor.fetchone()[0]

    if settings_count == 0:
        # Try to import from settings.json
        settings_json_path = Path(__file__).parent / "settings.json"
        if settings_json_path.exists():
            print("Migrating settings from settings.json to database...")
            conn.close()  # Close before import (import_settings_from_json opens its own connection)
            if import_settings_from_json(str(settings_json_path)):
                print("Settings migrated successfully from settings.json!")
            else:
                print("Warning: Failed to migrate settings from settings.json")
            return  # Already closed connection
        else:
            print("No settings.json found - starting with empty settings")

    conn.close()


# Initialize database on import
if not DB_PATH.exists():
    init_db()
else:
    migrate_db()

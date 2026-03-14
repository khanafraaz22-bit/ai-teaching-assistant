"""
Email Service
-------------
Sends transactional emails using SMTP.
If SMTP is not configured, emails are logged to console for development.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.config.settings import settings

logger = logging.getLogger(__name__)


class EmailService:

    def send_verification_email(self, to_email: str, token: str) -> None:
        verify_url = f"http://localhost:8000/api/v1/auth/verify-email?token={token}"
        subject = "Verify your YT Course Assistant account"
        html = f"<p>Verify here: <a href='{verify_url}'>{verify_url}</a></p>"

        if not settings.SMTP_HOST:
            print("\n" + "="*70)
            print("📧 EMAIL (dev mode — SMTP not configured)")
            print(f"To:      {to_email}")
            print(f"Subject: {subject}")
            print(f"")
            print(f"👉 VERIFICATION URL:")
            print(f"   {verify_url}")
            print(f"")
            print(f"👉 TOKEN ONLY (for Swagger):")
            print(f"   {token}")
            print("="*70 + "\n")
            return

        self._send(to_email, subject, html)

    def send_password_reset_email(self, to_email: str, token: str) -> None:
        reset_url = f"http://localhost:3000/reset-password?token={token}"
        subject = "Reset your YT Course Assistant password"
        html = f"<p>Reset here: <a href='{reset_url}'>{reset_url}</a></p>"

        if not settings.SMTP_HOST:
            print("\n" + "="*70)
            print("📧 EMAIL (dev mode — SMTP not configured)")
            print(f"To:      {to_email}")
            print(f"Subject: {subject}")
            print(f"")
            print(f"👉 PASSWORD RESET URL:")
            print(f"   {reset_url}")
            print(f"")
            print(f"👉 TOKEN ONLY (for Swagger):")
            print(f"   {token}")
            print("="*70 + "\n")
            return

        self._send(to_email, subject, html)

    def _send(self, to_email: str, subject: str, html: str) -> None:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = settings.EMAILS_FROM
        msg["To"]      = to_email
        msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                server.ehlo()
                server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(settings.EMAILS_FROM, to_email, msg.as_string())
            logger.info(f"Email sent to {to_email}: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            raise RuntimeError(f"Email delivery failed: {e}")
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List
from src.utils.config import SMTP_SERVER, SMTP_PORT, EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER

@dataclass
class Alert:
    """Dataclass to hold alert information."""
    severity: str
    title: str
    message: str
    company: str
    timestamp: datetime = field(default_factory=datetime.now)

class AlertSystem:
    """
    An advanced, rule-based alert system with daily summaries.
    """
    def __init__(self):
        self.alerts_log: List[Alert] = []
        self.alert_rules = {
            'health_score_critical': {'threshold': 40, 'severity': 'critical'},
            'health_score_warning': {'threshold': 55, 'severity': 'medium'},
            'price_drop_high': {'threshold': -5.0, 'severity': 'high'},
            'volume_spike': {'threshold': 3.0, 'severity': 'medium'},
            'negative_sentiment_cluster': {'threshold': -0.4, 'severity': 'high'}
        }

    def check_all_conditions(self, company_ticker, analysis_results, market_data):
        """Checks all data against predefined alert rules."""
        health_score = analysis_results.get("health_score", 70)
        news_sentiment = analysis_results.get("news_sentiment", 0)

        if health_score < self.alert_rules['health_score_critical']['threshold']:
            self._trigger_alert('critical', f"CRITICAL Health Score for {company_ticker}", f"Fundamental Health Score dropped to {health_score:.1f}", company_ticker)
        elif health_score < self.alert_rules['health_score_warning']['threshold']:
            self._trigger_alert('medium', f"Low Health Score for {company_ticker}", f"Fundamental Health Score is {health_score:.1f}", company_ticker)
        
        if market_data is not None and not market_data.empty and len(market_data) > 1:
            latest = market_data.iloc[-1]
            price_change_pct = ((latest['Close'] - market_data.iloc[-2]['Close']) / market_data.iloc[-2]['Close']) * 100
            if price_change_pct < self.alert_rules['price_drop_high']['threshold']:
                 self._trigger_alert('high', f"Significant Price Drop for {company_ticker}", f"Stock price dropped by {price_change_pct:.2f}%", company_ticker)
            if latest.get('Volume_Ratio', 0) > self.alert_rules['volume_spike']['threshold']:
                self._trigger_alert('medium', f"Unusual Volume Spike for {company_ticker}", f"Trading volume is {latest.get('Volume_Ratio', 0):.1f}x the 20-day average.", company_ticker)

        if news_sentiment < self.alert_rules['negative_sentiment_cluster']['threshold']:
            self._trigger_alert('high', f"Cluster of Negative News for {company_ticker}", f"Average news sentiment has dropped to {news_sentiment:.2f}", company_ticker)

    def _trigger_alert(self, severity, title, message, company):
        alert = Alert(severity=severity, title=title, message=message, company=company)
        self.alerts_log.append(alert)
        print(f"ALERT TRIGGERED: {alert}")
        if severity in ['critical', 'high']:
            self.send_email(
                subject=f"[{alert.severity.upper()}] Finance AI Alert: {alert.title}",
                body=self._format_alert_body(alert)
            )

    def _format_alert_body(self, alert: Alert):
        return f"Alert for {alert.company}:\n\n{alert.message}\n\nSeverity: {alert.severity.upper()}\nTime: {alert.timestamp}"

    def send_daily_summary(self):
        """Compiles and sends a summary of medium and low severity alerts from the last 24 hours."""
        cutoff = datetime.now() - timedelta(days=1)
        recent_alerts = [a for a in self.alerts_log if a.timestamp >= cutoff and a.severity in ['medium', 'low']]
        
        if not recent_alerts:
            print("No new medium/low alerts for the daily summary.")
            return

        summary_body = f"📊 Daily Corporate Intelligence Alert Summary - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        for alert in recent_alerts:
            summary_body += f"--- {alert.severity.upper()} ---\n"
            summary_body += f"Company: {alert.company}\n"
            summary_body += f"Title: {alert.title}\n"
            summary_body += f"Message: {alert.message}\n\n"
        
        self.send_email(
            subject=f"Daily Alert Summary for {datetime.now().strftime('%Y-%m-%d')}",
            body=summary_body
        )
        self.alerts_log = [a for a in self.alerts_log if a.timestamp < cutoff]

    def send_email(self, subject, body):
        """Generic email sending function."""
        if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
            print("Email credentials not configured. Skipping email.")
            return

        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(msg)
            print(f"Successfully sent email: '{subject}'")
        except Exception as e:
            print(f"Failed to send email: {e}")

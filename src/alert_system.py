import smtplib
from email.mime.text import MIMEText

def send_alert_email(subject, body, to_email):
    """Send an alert email."""
    from_email = "your_email@example.com"
    password = "your_password"  # Use an app-specific password if 2FA is enabled

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP('smtp.example.com', 587) as server:  # Replace with your email provider
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
            print(f"Alert email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def trigger_alert(condition):
    """Trigger an alert based on a specific condition."""
    if condition:  # Modify this condition based on your requirements
        subject = "Alert: Unusual Activity Detected"
        body = "Unusual activity has been detected in the monitored area."
        send_alert_email(subject, body, "recipient@example.com")

if _name_ == "_main_":
    # Example condition check
    unusual_activity_detected = True  # Replace with your logic
    trigger_alert(unusual_activity_detected)
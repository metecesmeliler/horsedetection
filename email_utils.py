import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time


def send_email(receiver, html_content, plain_content, subject):
    sender = "narin59a@gmail.com"

    # Create a MIMEMultipart message to include both plain text and HTML versions
    email_message = MIMEMultipart("alternative")
    email_message["Subject"] = subject
    email_message["From"] = sender
    email_message["To"] = receiver

    # Create MIMEText parts for plain text and HTML content
    plain_part = MIMEText(plain_content)
    html_part = MIMEText(html_content, "html")

    # Attach the parts to the message
    email_message.attach(plain_part)
    email_message.attach(html_part)

    try:
        with smtplib.SMTP("sandbox.smtp.mailtrap.io", 2525) as server:
            server.starttls()
            server.login("1cc82da90f40df", "08903a574d3834")
            server.sendmail(sender, receiver, email_message.as_string())
    except Exception as e:
        print("Error sending email:", e)

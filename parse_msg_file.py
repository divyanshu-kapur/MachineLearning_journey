import extract_msg

# Path to the .msg file
msg_file_path = "MachineLearning_journey/Tabcorp Data Domain Tier Utilization.msg"

# Function to parse the .msg file
def parse_msg_file(file_path):
    msg = extract_msg.Message(file_path)
    msg_content = {
        "subject": msg.subject,
        "date": msg.date,
        "body": msg.body,
        "html_body": msg.htmlBody,
        "attachments": [att.filename for att in msg.attachments]
    }
    return msg_content

# Parse the .msg file
msg_content = parse_msg_file(msg_file_path)

# Print the extracted content
print("Subject:", msg_content["subject"])
print("Date:", msg_content["date"])
print("Body:", msg_content["body"])
print("HTML Body:", msg_content["html_body"])

# Save the body to a text file for further processing
with open("parsed_email_body.txt", "w") as f:
    f.write(msg_content["body"])

if __name__ == "__main__":
    msg_content = parse_msg_file(msg_file_path)
    print("Subject:", msg_content["subject"])
    print("Date:", msg_content["date"])
    print("Body:", msg_content["body"])
    print("HTML Body:", msg_content["html_body"])

# Cybersecurity Red Team and Bug Bounty Chatbot

This project is an advanced chatbot designed to provide information about cybersecurity red team operations and bug bounty programs. It uses state-of-the-art natural language processing and machine learning techniques to offer accurate and helpful responses.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Data Preparation](#data-preparation)
5. [Running the Chatbot](#running-the-chatbot)
6. [Using the Chatbot](#using-the-chatbot)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [License](#license)

## Prerequisites

Before setting up the chatbot, ensure you have the following:

- Python 3.8 or higher
- pip (Python package manager)
- Git
- A valid OpenAI API key
- (Optional) Google OAuth credentials for authentication

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/DudeNation/chatbot-ML.git
   cd chatbot-ML
   ```

2. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip3 install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the project root and add the following credentials:

### OpenAI API Key
```
OPENAI_API_KEY=your_api_key_here
```
To obtain your OpenAI API key:
1. Visit https://platform.openai.com/account/api-keys
2. Sign up or log in to your OpenAI account
3. Click on "Create new secret key"
4. Copy the generated key (Note: it will only be shown once)

### Google OAuth Credentials
```
OAUTH_GOOGLE_CLIENT_ID=your_client_id_here
OAUTH_GOOGLE_CLIENT_SECRET=your_client_secret_here
```
To get Google OAuth credentials:
1. Go to https://console.cloud.google.com/
2. Create a new project or select an existing one
3. Enable the Google OAuth2 API
4. Go to "Credentials" → "Create Credentials" → "OAuth client ID"
5. Configure the OAuth consent screen
6. Select "Web application" as the application type
7. Add authorized redirect URIs (e.g., `http://localhost:8000/oauth/callback`)
8. Copy the generated Client ID and Client Secret

### Chainlit Authentication Secret
```
CHAINLIT_AUTH_SECRET=your_random_secret_here
```
Generate a secure random secret using Python:
```python
import secrets
print(secrets.token_urlsafe(32))
```

### Literal API Key
```
LITERAL_API_KEY=your_literal_api_key_here
```
To obtain your Literal API key:
1. Visit https://literal.ai/
2. Create an account or sign in
3. Navigate to your account settings
4. Generate a new API key from the API section

### Discord Bot Token and Channel IDs
```
DISCORD_BOT_TOKEN=your_discord_token_here
ALLOWED_CHANNEL_IDS=your_channel_id_here
```
To get your Discord bot token:
1. Go to https://discord.com/developers/applications
2. Click "New Application" and give it a name
3. Go to the "Bot" section and click "Add Bot"
4. Click "Reset Token" to reveal your bot token
5. Enable necessary Privileged Gateway Intents

To get channel IDs:
1. Enable Developer Mode in Discord (User Settings → App Settings → Advanced)
2. Right-click on a channel and select "Copy ID"
3. Multiple channel IDs can be comma-separated

### Security Notes:
- Never commit your `.env` file to version control
- Regularly rotate your API keys and secrets
- Use appropriate scopes and permissions for OAuth credentials
- Keep your Discord bot token private and secure
- Consider using a secrets management service for production deployments

## Data Preparation

The chatbot relies on HTML files containing cybersecurity information. To gather this data:

1. Create a `blogs.txt` file with URLs of cybersecurity blogs you want to include.

2. Run the `save_webpage.py` script to download and save the blog content:
   ```
   python3 save_webpage.py
   ```

   This script will:
   - Read URLs from `blogs.txt`
   - Download the content of each webpage
   - Save the HTML files in the `data` folder

3. The script uses Selenium WebDriver, so make sure you have Chrome installed and the appropriate ChromeDriver in your system PATH.

## Running the Chatbot

There are two ways to run the chatbot:

### 1. Server Terminal (Command-line interface)

To run the chatbot in the terminal:

```
python3 chatbot.py
```

This will start the chatbot in the command-line interface, allowing you to interact with it directly in the terminal.

### 2. Chainlit Web Interface

For a more user-friendly experience with a web-based UI:

```
chainlit run chainlit_app.py -w
```

This command will start the Chainlit server and open a web interface in your default browser.

## Using the Chatbot

1. Once the chatbot is running, you can start asking questions about cybersecurity red team operations and bug bounty programs.

2. The chatbot supports multi-modal interactions. You can upload images related to cybersecurity, and the chatbot will analyze them and incorporate the information into its responses.

3. To end a chat session, simply type "exit" or close the browser tab if using the web interface.

## Troubleshooting

- If you encounter any issues with package installations, try upgrading requirements:
  ```
  pip3 install --upgrade -r requirements.txt
  ```

- Make sure all the required environment variables are set correctly in the `.env` file.

- If the chatbot is slow to respond, check your internet connection and OpenAI API status.

- For issues with image analysis, ensure you have a stable internet connection and that your OpenAI API key has access to the required models.

## Contributing

Contributions to improve the chatbot are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
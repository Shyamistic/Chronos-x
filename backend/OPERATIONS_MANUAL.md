# 🛠️ ChronosX Operations Manual

This document outlines procedures for administering and managing the live ChronosX trading system.

---

## 1. Dynamic Configuration Hot-Reload

The system supports real-time configuration updates without requiring a server restart. This is the primary method for tuning risk parameters during live competition.

### Procedure

1.  **Modify Configuration:**
    *   SSH into the EC2 instance.
    *   Open the configuration file: `vim backend/config.py`.
    *   Change any desired parameter (e.g., `MIN_CONFIDENCE`, `HARDSTOP_PCT`, `MAX_POSITION_SIZE`).
    *   Save and close the file.

2.  **Set Admin Password (If not in `.env`):**
    *   Ensure the `.env` file in the project root (`/root/chronosx/.env`) contains the `ADMIN_PASSWORD`.
    *   Example line in `.env`: `ADMIN_PASSWORD="your_secret_password"`

3.  **Trigger the Hot-Reload:**
    *   From your **local machine** (or any machine with `curl`), execute a `POST` request to the `/api/admin/reload-config` endpoint.
    *   You must provide the admin password as a query parameter.

    ```powershell
    # Example using PowerShell/curl
    curl.exe -X POST "http://<YOUR_SERVER_IP>/api/admin/reload-config?password=<YOUR_ADMIN_PASSWORD>"
    ```

4.  **Verify the Change:**
    *   The API will return a `{"status":"success",...}` message.
    *   Check the live server logs to confirm the reload:
        ```bash
        journalctl -u chronosx -f
        ```
    *   You will see a "CONFIGURATION HOT-RELOADED" banner printed with the newly active parameters. The system will immediately start using these new rules for all subsequent trades.

---

## 2. Restarting the Service

If you need to perform a full restart after pulling new code, use `systemctl`.

```bash
sudo systemctl restart chronosx
```
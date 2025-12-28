# Deployment Guide

ChronosX is deployed as a production‑grade FastAPI backend behind nginx on AWS EC2, following modern best practices for performance and security. [web:7][web:19]

---

## High‑Level Topology

[ Internet ]
│
[ nginx ] ← serves frontend + proxies /api/*
│
[ FastAPI (ChronosX Core, uvicorn) ]
│
[ Exchange (WEEX) ]

text

Key properties:

- The FastAPI backend listens on `127.0.0.1:8000` only  
- nginx is the only public‑facing HTTP entry point  
- No client‑side secrets are shipped in the frontend bundle [web:7]

---

## Backend (FastAPI + uvicorn)

Core command:

uvicorn backend.api.main:app
--host 0.0.0.0
--port 8000

text

Recommended:

- Run under `systemd` as a managed service  
- Use a dedicated Unix user for the application  
- Configure resource limits and restart policies [web:19]

Example `systemd` unit (simplified):

[Unit]
Description=ChronosX FastAPI service
After=network.target

[Service]
User=chronosx
Group=chronosx
WorkingDirectory=/opt/chronosx
ExecStart=/usr/bin/env uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target

text

---

## Frontend (nginx Static Files)

- Build or place static frontend files under `/var/www/chronosx` (or similar)  
- nginx serves these files and proxies API requests to the backend [web:7][web:13]

Example nginx server block (simplified):

server {
listen 80;
server_name <public-ip-or-domain>;

text
root /var/www/chronosx;
index index.html;

location /api/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header   Host $host;
    proxy_set_header   X-Real-IP $remote_addr;
    proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto $scheme;
}

location / {
    try_files $uri /index.html;
}
}

text

Reload nginx after configuration changes:

sudo nginx -t
sudo systemctl reload nginx

text

---

## Hosting (AWS EC2)

Typical setup:

- EC2 instance (e.g., Ubuntu LTS)  
- Security group with:  
  - Port 22 (SSH) restricted to maintainers  
  - Port 80/443 (HTTP/HTTPS) open for users  
  - No public access to port 8000 [web:7][web:13]

ChronosX backend and exchange credentials live only on the backend host, never in the frontend.

---

## Security Considerations

- No public access to internal ports (`127.0.0.1:8000` only)  
- No credentials or API keys embedded in client‑side code  
- TLS termination recommended via nginx or a cloud load balancer  
- Principle of least privilege for system users and IAM roles [web:16][web:19]

This deployment matches typical production patterns for FastAPI systems and establishes a solid base for further hardening (WAF, rate limiting, CI/CD, etc.).

---
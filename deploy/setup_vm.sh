#!/bin/bash
# Instala Trentren en una VM Debian 12 (e2-micro). Uso: sudo bash setup_vm.sh <dominio>
set -euo pipefail
DOM="$1"
if ! swapon --show | grep -q swapfile; then
  fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv caddy >/dev/null
id trentren >/dev/null 2>&1 || useradd -r -m -d /opt/trentren -s /usr/sbin/nologin trentren
mkdir -p /opt/trentren && tar xzf /tmp/trentren.tgz -C /opt/trentren
[ -d /opt/trentren/.venv ] || python3 -m venv /opt/trentren/.venv
/opt/trentren/.venv/bin/pip install -q --upgrade pip
/opt/trentren/.venv/bin/pip install -q -r /opt/trentren/deploy/requirements.txt
chown -R trentren:trentren /opt/trentren && chmod 600 /opt/trentren/.env
cat > /etc/systemd/system/trentren.service <<UNIT
[Unit]
Description=Trentren - vigia sismico autonomo
After=network-online.target
[Service]
User=trentren
WorkingDirectory=/opt/trentren
ExecStart=/opt/trentren/.venv/bin/python -m trentren
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
[Install]
WantedBy=multi-user.target
UNIT
# La VM puede alojar otros sitios: se agrega el bloque de Trentren al Caddyfile sin tocar el resto.
if ! grep -q "^$DOM {" /etc/caddy/Caddyfile 2>/dev/null; then
  [ -f /etc/caddy/Caddyfile ] && cp /etc/caddy/Caddyfile "/etc/caddy/Caddyfile.bak-$(date +%Y%m%d-%H%M%S)"
  cat >> /etc/caddy/Caddyfile <<CADDY

$DOM {
	reverse_proxy 127.0.0.1:8787 {
		flush_interval -1
	}
}
CADDY
fi
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
systemctl daemon-reload && systemctl enable --now trentren && systemctl restart trentren
systemctl reload caddy
echo "listo: https://$DOM"

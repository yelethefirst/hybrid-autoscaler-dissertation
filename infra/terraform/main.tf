# Measurement environment (§3.10-A deviation: Hetzner CCX43 replaces AWS m6a.4xlarge
# — see docs/deviations.md. Equivalent spec: 16 vCPU AMD EPYC dedicated, 64 GiB RAM).

provider "hcloud" {
  token = var.hcloud_token
}

resource "hcloud_ssh_key" "measurement" {
  name       = var.name
  public_key = file(pathexpand(var.ssh_public_key_path))
}

resource "hcloud_firewall" "measurement" {
  name = var.name

  rule {
    description = "SSH from researcher IP only (§3.10-A: no external traffic to cluster NodePort)"
    direction   = "in"
    protocol    = "tcp"
    port        = "22"
    source_ips  = [var.allowed_ssh_cidr]
  }
}

resource "hcloud_server" "measurement" {
  name         = var.name
  server_type  = var.server_type
  image        = var.image
  location     = var.location
  ssh_keys     = [hcloud_ssh_key.measurement.id]
  firewall_ids = [hcloud_firewall.measurement.id]
  labels       = var.labels

  # cloud-init: create the ubuntu user so Ansible (and run-campaign.sh) can
  # connect as ubuntu — matching the AWS §3.10-A spec. Hetzner's default image
  # only has root; this copies root's authorised key to ubuntu with passwordless sudo.
  user_data = <<-EOT
    #cloud-config
    users:
      - name: ubuntu
        groups: [sudo]
        shell: /bin/bash
        sudo: ALL=(ALL) NOPASSWD:ALL
    runcmd:
      - mkdir -p /home/ubuntu/.ssh
      - cp /root/.ssh/authorized_keys /home/ubuntu/.ssh/authorized_keys
      - chown -R ubuntu:ubuntu /home/ubuntu/.ssh
      - chmod 700 /home/ubuntu/.ssh
      - chmod 600 /home/ubuntu/.ssh/authorized_keys
  EOT
}

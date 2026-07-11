output "server_id" {
  description = "Hetzner server ID."
  value       = hcloud_server.measurement.id
}

output "public_ip" {
  description = "Public IPv4 (static — does not change on reboot, only on delete/recreate)."
  value       = hcloud_server.measurement.ipv4_address
}

output "ssh_command" {
  description = "SSH command to reach the measurement host as ubuntu."
  value       = "ssh -i ${replace(var.ssh_public_key_path, ".pub", "")} ubuntu@${hcloud_server.measurement.ipv4_address}"
}

output "ansible_inventory_entry" {
  description = "Paste this line into infra/ansible/inventory.ini."
  value       = "measurement ansible_host=${hcloud_server.measurement.ipv4_address} ansible_user=ubuntu ansible_ssh_private_key_file=~/.ssh/id_ed25519"
}

output "delete_command" {
  description = <<-EOT
    Hetzner bills per hour of server existence (not runtime) — stopping saves nothing.
    Run this from your laptop after the campaign is complete to stop all billing.
  EOT
  value = "terraform destroy  # or: hcloud server delete ${var.name}"
}

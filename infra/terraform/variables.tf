# Every value that defines the measurement environment (§3.10-A deviation: Hetzner CCX43)
# has a committed default equal to the dissertation spec. Overriding any for a
# measured run is a deviation and must be recorded in docs/deviations.md.

variable "hcloud_token" {
  description = "Hetzner Cloud API token (read/write). Set in terraform.tfvars — never commit."
  type        = string
  sensitive   = true
}

variable "server_type" {
  description = <<-EOT
    Hetzner server type (§3.10-A deviation spec):
    ccx43 = 16 vCPU AMD EPYC dedicated / 64 GiB RAM / 360 GiB NVMe — €0.258/hr.
    Matches the AWS m6a.4xlarge core/RAM spec (original §3.10-A) with equivalent
    AMD EPYC microarchitecture. See docs/deviations.md.
  EOT
  type        = string
  default     = "ccx43"
}

variable "image" {
  description = "Hetzner OS image. Ubuntu 24.04 LTS amd64 matches §3.10-A OS spec."
  type        = string
  default     = "ubuntu-24.04"
}

variable "location" {
  description = "Hetzner datacenter (nbg1 = Nuremberg; lowest latency from UK, well-connected)."
  type        = string
  default     = "nbg1"
}

variable "ssh_public_key_path" {
  description = "Path to the SSH public key used to reach the host."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "allowed_ssh_cidr" {
  description = <<-EOT
    CIDR allowed to SSH (port 22). No default: set to your current public IP as
    <ip>/32 (e.g. via `curl -4s ifconfig.me`). SSH is the ONLY inbound port —
    the cluster NodePort (30080) stays local to the host (§3.10-A co-location).
  EOT
  type        = string

  validation {
    condition     = can(cidrnetmask(var.allowed_ssh_cidr))
    error_message = "allowed_ssh_cidr must be a valid IPv4 CIDR, e.g. 203.0.113.7/32 (curl -4s ifconfig.me)."
  }
}

variable "name" {
  description = "Server name and Hetzner label prefix."
  type        = string
  default     = "hybrid-autoscaler-measurement"
}

variable "labels" {
  description = "Hetzner resource labels (key/value strings only)."
  type        = map(string)
  default = {
    project = "hybrid-autoscaler-dissertation"
    purpose = "measurement-environment-3.10A"
    owner   = "omoyele-olabode"
  }
}

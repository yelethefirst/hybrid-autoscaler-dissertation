# Provider and Terraform version constraints.
# The generated .terraform.lock.hcl is COMMITTED so that `terraform init`
# resolves the exact same provider build everywhere (reproducibility pillar).

terraform {
  required_version = ">= 1.9"

  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "~> 1.49"
    }
  }
}

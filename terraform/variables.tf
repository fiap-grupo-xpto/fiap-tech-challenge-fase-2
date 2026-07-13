variable "aws_region" {
  type        = string
  description = "Região da AWS para provisionamento"
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Nome do projeto para identificação de recursos"
  default     = "fiap-tech-challenge-fase-2"
}

variable "environment" {
  type        = string
  description = "Ambiente de deployment"
  default     = "production"
}

variable "container_image_backend" {
  type        = string
  description = "Imagem Docker para o container do backend FastAPI"
  default     = "tech-challenge-backend:latest"
}

variable "container_image_frontend" {
  type        = string
  description = "Imagem Docker para o container do frontend Streamlit"
  default     = "tech-challenge-frontend:latest"
}

variable "gemini_api_key" {
  type        = string
  description = "Chave de API do Gemini para integração com LLM"
  sensitive   = true
}

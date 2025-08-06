{{/*
Expand the name of the chart.
*/}}
{{- define "gaudi3-scale.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "gaudi3-scale.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "gaudi3-scale.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "gaudi3-scale.labels" -}}
helm.sh/chart: {{ include "gaudi3-scale.chart" . }}
{{ include "gaudi3-scale.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "gaudi3-scale.selectorLabels" -}}
app.kubernetes.io/name: {{ include "gaudi3-scale.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "gaudi3-scale.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "gaudi3-scale.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Generate the full image name
*/}}
{{- define "gaudi3-scale.image" -}}
{{- $registry := default .Values.image.registry .Values.global.imageRegistry -}}
{{- $repository := .Values.image.repository -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- else -}}
{{- printf "%s:%s" $repository $tag -}}
{{- end -}}
{{- end }}

{{/*
Create the namespace name
*/}}
{{- define "gaudi3-scale.namespace" -}}
{{- default .Release.Namespace .Values.global.namespace }}
{{- end }}

{{/*
Create storage class name
*/}}
{{- define "gaudi3-scale.storageClass" -}}
{{- default .Values.global.storageClass .Values.storage.storageClass }}
{{- end }}

{{/*
Generate database URL
*/}}
{{- define "gaudi3-scale.databaseUrl" -}}
{{- if .Values.database.enabled -}}
{{- printf "postgresql://%s:%s@%s:%d/%s" .Values.database.user .Values.secrets.databasePassword .Values.database.host .Values.database.port .Values.database.name -}}
{{- else -}}
{{- .Values.secrets.databaseUrl -}}
{{- end -}}
{{- end }}

{{/*
Generate Redis URL
*/}}
{{- define "gaudi3-scale.redisUrl" -}}
{{- if .Values.redis.enabled -}}
{{- if .Values.secrets.redisPassword -}}
{{- printf "redis://:%s@%s:%d/%d" .Values.secrets.redisPassword .Values.redis.host .Values.redis.port .Values.redis.database -}}
{{- else -}}
{{- printf "redis://%s:%d/%d" .Values.redis.host .Values.redis.port .Values.redis.database -}}
{{- end -}}
{{- else -}}
{{- .Values.secrets.redisUrl -}}
{{- end -}}
{{- end }}

{{/*
Validate required values
*/}}
{{- define "gaudi3-scale.validateValues" -}}
{{- if and .Values.api.enabled (not .Values.api.replicaCount) -}}
{{- fail "API replica count must be specified when API is enabled" -}}
{{- end -}}
{{- if and .Values.trainer.enabled (not .Values.trainer.replicaCount) -}}
{{- fail "Trainer replica count must be specified when trainer is enabled" -}}
{{- end -}}
{{- if and .Values.storage.data.enabled (not .Values.storage.data.size) -}}
{{- fail "Data storage size must be specified when data storage is enabled" -}}
{{- end -}}
{{- end -}}
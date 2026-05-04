import { useEffect, useMemo, useState } from 'react'
import './App.css'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

const FIELD_CONFIG = [
  { key: 'Age', label: 'Age', unit: 'years', step: 1, decimals: 0 },
  { key: 'Height', label: 'Height', unit: 'cm', step: 0.1, decimals: 1 },
  { key: 'Weight', label: 'Weight', unit: 'kg', step: 0.1, decimals: 1 },
  { key: 'Duration', label: 'Duration', unit: 'min', step: 1, decimals: 0 },
  { key: 'Heart_Rate', label: 'Heart rate', unit: 'bpm', step: 1, decimals: 0 },
  { key: 'Body_Temp', label: 'Body temp', unit: 'C', step: 0.1, decimals: 1 },
]

const emptyForm = FIELD_CONFIG.reduce(
  (acc, field) => {
    acc[field.key] = ''
    return acc
  },
  { Sex: '' },
)

const clamp = (value, min, max) => Math.min(max, Math.max(min, value))

const formatValue = (value, decimals) => {
  if (!Number.isFinite(value)) return ''
  if (decimals === 0) return String(Math.round(value))
  return value.toFixed(decimals)
}

const buildInitialValues = (metadata) => {
  if (!metadata) return emptyForm
  const values = { Sex: metadata.sex_classes?.[0] || 'female' }
  FIELD_CONFIG.forEach((field) => {
    const range = metadata.feature_ranges?.[field.key]
    if (!range) {
      values[field.key] = ''
      return
    }
    const midpoint = clamp((range.min + range.max) / 2, range.min, range.max)
    values[field.key] = formatValue(midpoint, field.decimals)
  })
  return values
}

const buildUrl = (path) => (API_BASE ? `${API_BASE}${path}` : path)

function App() {
  const [metadata, setMetadata] = useState(null)
  const [metaStatus, setMetaStatus] = useState('loading')
  const [formValues, setFormValues] = useState(emptyForm)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        setMetaStatus('loading')
        const response = await fetch(buildUrl('/metadata'))
        if (!response.ok) {
          throw new Error('Unable to load metadata.')
        }
        const data = await response.json()
        setMetadata(data)
        setMetaStatus('ready')
      } catch (err) {
        setMetaStatus('error')
        setError('Unable to load model metadata. Try again later.')
      }
    }
    fetchMetadata()
  }, [])

  useEffect(() => {
    if (metadata) {
      setFormValues(buildInitialValues(metadata))
    }
  }, [metadata])

  const sexOptions = useMemo(() => {
    if (!metadata?.sex_classes?.length) return ['female', 'male']
    return metadata.sex_classes
  }, [metadata])

  const handleChange = (key) => (event) => {
    setFormValues((prev) => ({ ...prev, [key]: event.target.value }))
  }

  const validate = () => {
    if (!metadata) return 'Metadata is not available.'
    if (!formValues.Sex) return 'Select a sex option.'
    for (const field of FIELD_CONFIG) {
      const raw = formValues[field.key]
      const numeric = Number(raw)
      if (!Number.isFinite(numeric)) {
        return `Enter a valid ${field.label.toLowerCase()}.`
      }
      const range = metadata.feature_ranges?.[field.key]
      if (range && (numeric < range.min || numeric > range.max)) {
        return `${field.label} must be between ${formatValue(
          range.min,
          field.decimals,
        )} and ${formatValue(range.max, field.decimals)}.`
      }
    }
    return ''
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError('')
    setResult(null)

    const validationError = validate()
    if (validationError) {
      setError(validationError)
      return
    }

    const payload = { Sex: formValues.Sex }
    FIELD_CONFIG.forEach((field) => {
      payload[field.key] = Number(formValues[field.key])
    })

    try {
      setStatus('loading')
      const response = await fetch(buildUrl('/predict'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        const message = data?.detail || 'Prediction failed.'
        throw new Error(message)
      }
      const data = await response.json()
      setResult(data.calories)
      setStatus('success')
    } catch (err) {
      setStatus('idle')
      setError(err?.message || 'Prediction failed.')
    }
  }

  const handleReset = () => {
    setError('')
    setResult(null)
    setFormValues(buildInitialValues(metadata))
  }

  return (
    <div className="page">
      <header className="hero">
        <span className="eyebrow">Assessment demo</span>
        <h1>Calorie Predictor</h1>
        <p>
          Estimate calories burned from workout metrics. Input ranges are
          derived from the training data for more reliable predictions.
        </p>
        <div className="hero-badges">
          <span className="badge">Model: XGBoost</span>
          <span className="badge">Cross terms: 15</span>
          <span className="badge">Output: calories</span>
        </div>
      </header>

      <main className="content-grid">
        <section className="panel form-panel">
          <div className="panel-header">
            <h2>Input details</h2>
            <p>Stay within the learned ranges for best accuracy.</p>
          </div>

          {metaStatus === 'loading' && (
            <p className="inline-note">Loading ranges from the model...</p>
          )}
          {metaStatus === 'error' && (
            <p className="form-error">Unable to load metadata.</p>
          )}

          <form onSubmit={handleSubmit} className="form">
            <div className="field-grid">
              <div className="field">
                <label htmlFor="Sex">Sex</label>
                <select
                  id="Sex"
                  name="Sex"
                  value={formValues.Sex}
                  onChange={handleChange('Sex')}
                  required
                >
                  <option value="" disabled>
                    Select
                  </option>
                  {sexOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
                <span className="field-hint">
                  Classes: {sexOptions.join(', ')}
                </span>
              </div>

              {FIELD_CONFIG.map((field) => {
                const range = metadata?.feature_ranges?.[field.key]
                const rangeText = range
                  ? `Range: ${formatValue(range.min, field.decimals)} - ${formatValue(
                      range.max,
                      field.decimals,
                    )} ${field.unit}`
                  : 'Range pending'

                return (
                  <div className="field" key={field.key}>
                    <label htmlFor={field.key}>{field.label}</label>
                    <input
                      id={field.key}
                      name={field.key}
                      type="number"
                      inputMode="decimal"
                      step={field.step}
                      min={range?.min}
                      max={range?.max}
                      value={formValues[field.key]}
                      onChange={handleChange(field.key)}
                      required
                    />
                    <span className="field-hint">{rangeText}</span>
                  </div>
                )
              })}
            </div>

            <div className="actions">
              <button
                type="submit"
                disabled={status === 'loading' || metaStatus !== 'ready'}
              >
                {status === 'loading' ? 'Predicting...' : 'Predict calories'}
              </button>
              <button type="button" className="ghost" onClick={handleReset}>
                Reset to midpoints
              </button>
            </div>

            {error && <p className="form-error">{error}</p>}
          </form>
        </section>

        <aside className="panel result-panel">
          <div className="panel-header">
            <h2>Prediction</h2>
            <p>Instant estimate based on the trained model.</p>
          </div>

          <div className="result-card">
            <div className="result-value">
              {result !== null ? result.toFixed(1) : '--'}
            </div>
            <div className="result-unit">calories</div>
          </div>

          <div className="result-meta">
            <div>
              <h3>Model bounds</h3>
              <p>
                {metadata
                  ? `${formatValue(metadata.clip_min, 1)} - ${formatValue(
                      metadata.clip_max,
                      1,
                    )}`
                  : 'Loading...'}
              </p>
            </div>
            <div>
              <h3>Input ranges</h3>
              <p>Derived from the training data.</p>
            </div>
          </div>
        </aside>
      </main>

      <footer className="footer">
        Predictions outside the trained ranges may be less reliable.
      </footer>
    </div>
  )
}

export default App

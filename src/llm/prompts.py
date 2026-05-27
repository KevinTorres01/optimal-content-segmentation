COHESION_SCORING_PROMPT = """Evalúa la cohesión semántica de este segmento de texto en una escala del 1 al 5:
- 1: Sin coherencia. Las oraciones hablan de temas completamente distintos.
- 2: Coherencia baja. Hay temas relacionados pero las transiciones son muy bruscas.
- 3: Coherencia parcial. Temas relacionados pero el hilo narrativo es irregular.
- 4: Buena coherencia. Las oraciones se complementan bien con pequeñas interrupciones.
- 5: Coherencia perfecta. Todas las oraciones contribuyen a un único tema claro.

Segmento de texto:
{segment_text}

Responde ÚNICAMENTE con un objeto JSON válido (sin markdown, sin texto adicional):
{{"score": <entero del 1 al 5>, "rationale": "<una oración que explique la puntuación>"}}"""

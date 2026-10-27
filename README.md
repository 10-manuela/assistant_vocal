Ce projet consiste en la création d'un agent vocal autonome (voix-à-voix) qui fonctionne entièrement sur la machine de l'utilisateur, sans aucune dépendance au cloud ni connexion internet. Il combine une détection de silence intelligente, un traitement de langage naturel de pointe, une analyse biométrique de la voix et une interface graphique dynamique inspirée des films de science-fiction.

Fonctionnalités Clés :

• Pipeline Vocal Intégral (Speech-to-Speech) : Un enchaînement fluide et ultra-rapide qui écoute la voix humaine, la transcrit textuellement (Faster-Whisper), génère une réponse contextuelle (Ollama/Llama 3.2) et la verbalise à l'oral en moins de deux secondes.

• Biométrie Vocale & Adaptation : À chaque interaction, le système applique une analyse spectrale par transformation de Fourier (FFT) sur le signal audio pour évaluer la fréquence fondamentale de la voix. Il détecte ainsi instantanément le genre de l'interlocuteur pour adapter ses formules de politesse ("Monsieur" ou "Madame").

• Cerveau Contextuel Local : Propulsé par Ollama, l'assistant possède une mémoire de conversation active lui permettant de se souvenir des répliques précédentes et de maintenir un dialogue naturel et cohérent.

• Interface Holographique Cinétique : Développée en CustomTkinter, l'interface arbore un look futuriste bleu électrique. Des anneaux concentriques tournent en sens inverses de manière asynchrone, pulsent en mode attente, et vibrent dynamiquement au rythme exact de la voix lorsque l'IA s'exprime.

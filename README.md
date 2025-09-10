# 📚 Naïve RAG (Retrieval-Augmented Generation)

## 🚀 Introduction
Ce projet implémente un **pipeline RAG standard (aussi appelé RAG naïf)**.  
Il s’agit de l’architecture de base pour connecter un **modèle de langage (LLM)** à une **base documentaire**, afin de générer des réponses **plus fiables et contextualisées** que celles d’un LLM seul.  

Contrairement aux variantes avancées (RAG agentique, RAG hybride), ce pipeline est **entièrement séquentiel** :  
- il prend les documents bruts,  
- les transforme en représentations vectorielles,  
- effectue une recherche dense,  
- et injecte le contexte dans le modèle pour générer une réponse.  

L’objectif est pédagogique : comprendre et implémenter les **fondations de RAG** avant d’explorer les améliorations possibles.  

---

## 🏗️ Étapes détaillées du Pipeline

### 1. Acquisition et Prétraitement des Données
- **Sources** : Documents PDF (scientifiques, rapports, manuels).  
- **Extraction** : bibliothèque [Unstructured](https://github.com/Unstructured-IO/unstructured), qui segmente automatiquement le texte en blocs exploitables.  
- **Nettoyage (`extraction.py`)** :  
  - Suppression des métadonnées (ISBN, éditeurs, dépôts légaux).  
  - Élimination des numéros de page, tables des matières et en-têtes/pieds répétés.  
  - Correction des artefacts d’extraction (`!` → puces, suppression des doublons).  
- **Sortie** : un fichier consolidé `data/texte_nettoye.txt`.  

⚡ *Pourquoi ?*  
Un texte brut non nettoyé introduit du bruit → embeddings moins précis → réponses moins fiables.  

---

### 2. Segmentation en *Chunks*
- **Outil** : `RecursiveCharacterTextSplitter` (LangChain).  
- **Paramètres choisis** :  
  - Taille des chunks : `512 caractères`  
  - Chevauchement : `64 caractères`  
- **Pourquoi ces valeurs ?**  
  - 512 caractères ≈ 100 tokens → équilibre entre granularité et cohérence.  
  - Overlap = 64 → maintien du contexte entre segments.  
- **Sortie** : fichier JSON contenant les chunks + métadonnées (id, source).  

Exemple d’un chunk :  
```json
{
  "id": "chunk_0042",
  "source": "data/texte_nettoye.txt",
  "content": "Alan Turing est considéré comme le père de l’informatique moderne...",
  "start": 1500,
  "end": 2012
}
```
# 3. Génération des Représentations Vectorielles (Embeddings)

**Modèle utilisé :** `intfloat/multilingual-e5-base`  

- Encodeur multilingue performant  
- Spécialement adapté au **français** et à l’**anglais**  

## Caractéristiques
- Vecteurs de dimension **768**  
- Normalisation **L2** activée → mesure de similarité **cosinus** stable  
- **Script :** `embed.py`  
- **Sortie :** `chunks_with_embeddings.json`  

## ⚡ Pourquoi ce modèle ?
- **Léger (base model)** → rapide à exécuter  
- **Multilingue** → extensible à plusieurs langues  
- Bonne compatibilité avec **FAISS** et **Qdrant**  
# 4. Indexation Vectorielle

**Moteurs testés :**  
- `Qdrant` (par défaut)  
- `FAISS` (alternative locale)  

## Configuration Qdrant
- **Métrique de distance :** cosinus  
- Collection reconstruite automatiquement si existante  
- **Insertion par lots** (batch size = 200) → évite surcharge réseau  
- **Script :** `get_vector_db.py`  

## ⚡ Pourquoi Qdrant ?
- Support natif des recherches **scalables**  
- **API Python moderne**  
- Gestion robuste des **batchs**  

---

# 5. Transformation de la Requête Utilisateur

- La requête est encodée avec le **même modèle d’embedding**  
- Comparaison avec tous les embeddings stockés via **similarité cosinus**  

### Exemple de représentation (simplifiée) :
```python
query_vector = encoder.encode("Qui est Alan Turing?")
similarities = cosine_similarity(query_vector, stored_embeddings)
```
# 6. Recherche Dense et Récupération

- **Stratégie appliquée :** pure dense retrieval  
- Les **top-k** (par défaut k=5) chunks les plus proches sont sélectionnés  
- Pas de reranking lexical (**BM25**) ni de combinaison hybride  

## ⚡ Avantages
- Rapidité  

## ⚡ Limites
- Sensibilité aux erreurs d’**embedding**  

---

# 7. Fusion et Génération de la Réponse

- Concaténation des chunks sélectionnés  
- Injection dans un **LLM :** `qwen2.5` via **Ollama**  

## Stratégie de génération
- Réponse **concise**  
- Fidèle aux **documents sources**  

### Exemple :
**Input :**  
Qui est Alan Turing ?
**Output :**  
Alan Turing est un mathématicien britannique, pionnier de l’informatique et de la cryptanalyse.
Il est notamment connu pour sa contribution au décryptage d’Enigma et pour le test de Turing.
# 🧩 Architecture Globale

Le pipeline repose sur **3 couches principales** :  

1. **Ingestion & préparation des données**  
   - Extraction PDF, nettoyage, segmentation  

2. **Stockage & recherche vectorielle**  
   - Embeddings multilingues  
   - Qdrant / FAISS  

3. **Génération augmentée**  
   - Top-k retrieval → injection contexte → LLM  

## Diagramme du pipeline

```mermaid
graph TD
    A[📂 PDF Sources] --> B[🧹 Extraction & Nettoyage]
    B --> C[✂️ Chunking]
    C --> D[🔢 Embeddings]
    D --> E[🗄️ Vector DB (Qdrant/FAISS)]
    E --> F[❓ User Query]
    F --> G[🔍 Similarité Cosinus]
    G --> H[📑 Top-k Chunks]
    H --> I[🤖 LLM (qwen2.5 via Ollama)]
    I --> J[📝 Réponse Contextualisée]
```
# ⚙️ Installation & Exécution

## 1. Cloner le projet
```bash
git clone https://github.com/username/naive-rag.git
cd naive-rag
```
## 2.Installer les dépendances
```bash
pip install -r requirements.txt
```
## 3. Lancer une démo
```bash
python src/rag.py
```
# 📈 Limites actuelles
- Recherche **purement vectorielle** (pas de BM25 / reranking)  
- Pas de **filtrage dynamique** ni de **context re-ranking**  
- Pas de **self-evaluation** ou correction a posteriori  

---

# 🔮 Améliorations futures
- Intégrer un **retrieval hybride** (dense + lexical)  
- Utiliser un **LLM plus avancé** (Mistral, GPT, Llama 3)  
- Ajouter un **module de feedback utilisateur**  
- Implémenter un **RAG agentique** (capacité de raisonnement + actions externes)  

---

# 👩‍💻 Auteur
**Khaoula Boughattas** – Étudiante en **Data Engineering & Decision Systems**




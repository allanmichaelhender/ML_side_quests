import "./App.css";

interface Quest {
  id: number;
  title: string;
  description: string;
  emoji: string;
  tags: string[];
  streamlitUrl: string | null;
}

const QUESTS: Quest[] = [
  {
    id: 1,
    title: "Plant Disease Classification",
    description:
      "MobileNetV2-based classifier detecting 38 diseases across 14 crop species from leaf images.",
    emoji: "🌱",
    tags: ["CV", "MobileNetV2", "ONNX", "Grad-CAM"],
    streamlitUrl: "https://quest-01-cv-plant-disease.streamlit.app",
  },
  {
    id: 2,
    title: "Sentiment Analysis",
    description:
      "DistilBERT fine-tuned on Amazon reviews for 3-class sentiment classification with attention visualisation.",
    emoji: "📝",
    tags: ["NLP", "DistilBERT", "Hugging Face", "Attention"],
    streamlitUrl: "https://quest-02-nlp-sentiment-analysis.streamlit.app",
  },
  {
    id: 3,
    title: "Energy Earnings Call Analyst",
    description:
      "LoRA fine-tuned LLM extracting sentiment and structured claims from SEC EDGAR earnings transcripts.",
    emoji: "📊",
    tags: ["GenAI", "LoRA", "PEFT", "EDGAR"],
    streamlitUrl: null,
  },
  {
    id: 4,
    title: "RAG Retrieval Pipeline",
    description:
      "End-to-end retrieval evaluation comparing BM25, dense, hybrid, and reranked search over arXiv abstracts.",
    emoji: "🔍",
    tags: ["RAG", "FAISS", "Sentence-Transformers", "Reranking"],
    streamlitUrl: null,
  },
  {
    id: 5,
    title: "Support Ticket Routing",
    description:
      "TF-IDF, DistilBERT, and zero-shot LLM compared on multi-class ticket routing for customer support.",
    emoji: "🎫",
    tags: ["NLP", "Classification", "TF-IDF", "Zero-shot"],
    streamlitUrl: null,
  },
  {
    id: 6,
    title: "Energy Grid Load Balancing",
    description:
      "PPO reinforcement learning agent dispatching generation sources to balance cost, emissions, and reliability.",
    emoji: "⚡",
    tags: ["RL", "PPO", "Stable Baselines3", "Gymnasium"],
    streamlitUrl: "https://quest-06-rl-grid-balancing.streamlit.app",
  },
];

function QuestCard({ quest }: { quest: Quest }) {
  return (
    <a
      className={`quest-card ${quest.streamlitUrl ? "" : "coming-soon"}`}
      href={quest.streamlitUrl ?? undefined}
      target={quest.streamlitUrl ? "_blank" : undefined}
      rel={quest.streamlitUrl ? "noopener noreferrer" : undefined}
    >
      <div className="quest-emoji">{quest.emoji}</div>
      <h2 className="quest-title">{quest.title}</h2>
      <p className="quest-description">{quest.description}</p>
      <div className="quest-tags">
        {quest.tags.map((tag) => (
          <span key={tag} className="quest-tag">
            {tag}
          </span>
        ))}
      </div>
      {!quest.streamlitUrl && (
        <div className="coming-soon-badge">Coming Soon</div>
      )}
    </a>
  );
}

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>ML Side Quests</h1>
        <p className="app-subtitle">
          Six industry-focused ML projects spanning computer vision, NLP,
          generative AI, retrieval, and reinforcement learning.
        </p>
      </header>

      <main className="quest-grid">
        {QUESTS.map((quest) => (
          <QuestCard key={quest.id} quest={quest} />
        ))}
      </main>

      <footer className="app-footer">
        <p>
          Built with React + Vite &middot; Streamlit apps deployed on Streamlit
          Community Cloud
        </p>
      </footer>
    </div>
  );
}

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
      "MobileNetV2-based classifier detecting diseases across 3 crop species from leaf images.",
    emoji: "🌱",
    tags: ["CV", "MobileNetV2", "ONNX", "Grad-CAM"],
    streamlitUrl: "https://ml-sidequests-01.streamlit.app",
  },
  {
    id: 2,
    title: "Sentiment Analysis",
    description:
      "DistilBERT fine-tuned on Amazon reviews for 3-class sentiment classification with attention visualisation.",
    emoji: "📝",
    tags: ["NLP", "DistilBERT", "Hugging Face", "Attention"],
    streamlitUrl: "https://ml-sidequests-02.streamlit.app",
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
    streamlitUrl: "https://ml-sidequests-04.streamlit.app",
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
    streamlitUrl: "https://ml-sidequests-06.streamlit.app",
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
          Industry application focused ML projects spanning Computer Vision,
          NLP, LLM fine-tuning, RAG retrieval, and Reinforcement Learning.
        </p>
      </header>

      <main className="quest-grid">
        {QUESTS.map((quest) => (
          <QuestCard key={quest.id} quest={quest} />
        ))}
      </main>

      <footer className="app-footer">
        <p>Streamlit apps deployed on Streamlit Community Cloud</p>
        <p className="footer-repo">
          <a
            href="https://github.com/allanmichaelhender/ML_side_quests"
            target="_blank"
            rel="noopener noreferrer"
          >
            <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            GitHub Repository
          </a>
        </p>
      </footer>
    </div>
  );
}

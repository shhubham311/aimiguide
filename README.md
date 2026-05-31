# AI/ML Learning Guide

A comprehensive, interactive learning platform for mastering AI, Machine Learning, and related technologies. Built with modern web technologies to provide an engaging learning experience with structured roadmaps, project tracking, and interactive lessons.

## 🎯 Features

- **Interactive Learning Paths**: Structured curriculum covering foundational to advanced AI/ML concepts
- **Lesson Viewer**: Beautiful markdown-based lesson rendering with syntax highlighting
- **Learning Tracker**: Monitor your progress through different topics and courses
- **Project Tracker**: Track hands-on projects and learning milestones
- **Roadmap Dashboard**: Visual representation of your learning journey
- **Dark Mode Support**: Comfortable reading experience in any lighting condition
- **Responsive Design**: Seamless experience across desktop, tablet, and mobile devices
- **Code Highlighting**: Syntax highlighting for code snippets throughout lessons

## 📚 Curriculum Coverage

The platform covers comprehensive topics including:

### **Mathematics & Statistics**
- Probability Basics & Permutation-Combination
- Descriptive & Inferential Statistics
- Hypothesis Testing
- Linear Algebra (Matrices, Determinants, Eigenvalues, Decompositions)
- Calculus (Limits, Continuity, Optimization, Taylor Series)
- Vector Spaces & Correlation-Covariance

### **Python & DSA**
- Python Basics & OOPS
- Data Structures & Algorithms
- Common DSA Problems
- File I/O Operations
- SQL & Python Integration

### **Data Science & Analysis**
- NumPy Fundamentals
- Pandas & EDA
- Data Visualization
- ETL & AWS Deployment

### **Machine Learning**
- ML Introduction & Core Concepts
- Linear & Logistic Regression
- Decision Trees & Ensemble Methods
- KNN, Naive Bayes, SVM
- Clustering & PCA
- Time Series Analysis
- Evaluation Metrics & Model Testing
- Learning to Rank

### **Deep Learning**
- MLPs & Gradient Descent
- CNNs (Convolutional Neural Networks)
- RNNs, GRU, LSTM
- Transformers & BERT
- GANs (Generative Adversarial Networks)
- Diffusion Models

### **NLP & Advanced Topics**
- NLP Basics
- Advanced NLP & Embeddings
- Multimodal Embeddings
- Vision-Language Models
- LLMs & Fine-tuning
- Retrieval & Candidate Generation
- Search, Retrieval & Recommendation Systems

### **Deployment & Tools**
- FastAPI Development
- Database Fundamentals
- Deployment Types
- AWS Deployment
- Git & GitHub
- Interview Patterns & Important Concepts

## 🛠 Tech Stack

- **Frontend Framework**: [Next.js 16](https://nextjs.org/) with TypeScript
- **Styling**: [Tailwind CSS 4](https://tailwindcss.com/) with animations
- **UI Components**: [shadcn/ui](https://ui.shadcn.com/) with Radix UI
- **Markdown Rendering**: React Markdown with syntax highlighting
- **Code Highlighting**: [highlight.js](https://highlightjs.org/) & react-syntax-highlighter
- **Animation**: [Framer Motion](https://www.framer.com/motion/)
- **Theme Support**: [next-themes](https://github.com/pacocoursey/next-themes)
- **Notifications**: [Sonner](https://sonner.emilkowal.ski/) toast notifications
- **Validation**: [Zod](https://zod.dev/) schema validation
- **Command Palette**: [cmdk](https://cmdk.paco.sh/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Web SDK**: z-ai-web-dev-sdk for enhanced functionality

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ or npm/yarn
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/aimlguide.git
cd aimlguide
```

2. **Install dependencies**
```bash
npm install
```

3. **Run development server**
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

### Available Scripts

```bash
npm run dev      # Start development server on port 3000
npm run build    # Build for production
npm start        # Start production server
npm run lint     # Run ESLint for code quality
```

## 📁 Project Structure

```
aimlguide/
├── src/
│   ├── app/                          # Next.js app directory
│   │   ├── api/                      # API routes
│   │   │   ├── route.ts             # Main API endpoint
│   │   │   └── lesson/              # Lesson API endpoints
│   │   ├── layout.tsx               # Root layout
│   │   ├── page.tsx                 # Home page
│   │   └── globals.css              # Global styles
│   ├── components/                   # React components
│   │   ├── learning-tracker.tsx     # Progress tracking component
│   │   ├── lesson-viewer.tsx        # Lesson display component
│   │   ├── project-tracker.tsx      # Project management component
│   │   ├── roadmap-dashboard.tsx    # Learning roadmap visualization
│   │   └── ui/                      # shadcn/ui components
│   ├── content/                      # Markdown lesson content
│   │   ├── *.md                     # Individual lesson files
│   │   └── ...
│   ├── hooks/                        # Custom React hooks
│   │   ├── use-mobile.ts            # Mobile detection hook
│   │   └── use-toast.ts             # Toast notification hook
│   └── lib/                          # Utilities & helpers
│       ├── course-data.ts           # Course structure data
│       └── ...
├── public/                           # Static assets
├── examples/                         # Example code
│   └── websocket/                   # WebSocket examples
├── components.json                   # shadcn/ui config
├── next.config.ts                   # Next.js configuration
├── tailwind.config.ts               # Tailwind CSS configuration
├── tsconfig.json                    # TypeScript configuration
├── package.json                     # Dependencies & scripts
├── eslint.config.mjs                # ESLint configuration
├── postcss.config.mjs               # PostCSS configuration
└── Caddyfile                        # Web server configuration
```

## 🔌 API Reference

### GET /api
Returns a welcome message for the API.

**Response:**
```json
{
  "message": "Hello, world!"
}
```

### GET /api/lesson
Fetches lesson content and metadata.

**Query Parameters:**
- `id` (string): Lesson identifier
- `topic` (string): Topic/category of the lesson

**Response:**
```json
{
  "id": "string",
  "title": "string",
  "content": "string (markdown)",
  "metadata": {
    "difficulty": "string",
    "duration": "number",
    "prerequisites": "array"
  }
}
```

## 🎨 Customization

### Theming
The application supports light and dark modes. Theme preference is persisted to localStorage using `next-themes`.

### Styling
- Global styles: [src/app/globals.css](src/app/globals.css)
- Tailwind config: [tailwind.config.ts](tailwind.config.ts)
- Component themes: Component-level CSS modules

### Content
Add new lessons by creating `.md` files in the `src/content/` directory. The system automatically discovers and registers new content.

## 🚢 Deployment

### Deploy to Vercel

1. **Push to GitHub**
```bash
git add .
git commit -m "Ready for Vercel deployment"
git push origin main
```

2. **Connect to Vercel**
   - Visit [vercel.com](https://vercel.com)
   - Click "Add New Project"
   - Select your GitHub repository
   - Click "Import"

3. **Configure Settings** (auto-detected for Next.js)
   - Framework: Next.js
   - Build Command: `npm run build`
   - Output Directory: `.next`

4. **Deploy**
   - Click "Deploy"
   - Your app will be live in minutes!

### Environment Variables
If needed, add environment variables in Vercel dashboard:
- Settings → Environment Variables
- Add your variables and redeploy

### Manual Deployment
```bash
npm run build
npm start
```

## 📊 Performance Optimizations

- **Code Splitting**: Automatic code splitting via Next.js
- **Image Optimization**: Optimized image loading and rendering
- **Package Imports**: Optimized imports for `lucide-react`, `recharts`, `framer-motion`
- **Markdown Rendering**: Efficient markdown parsing and syntax highlighting
- **CSS**: Tailwind CSS with tree-shaking for minimal bundle size

## 🔧 Configuration Files

### `next.config.ts`
- TypeScript build error ignoring for development flexibility
- React Strict Mode disabled for smoother development
- Package import optimization

### `tsconfig.json`
- Strict TypeScript settings
- Path aliases for clean imports

### `tailwind.config.ts`
- Custom design tokens
- Animation presets
- Extended theme configurations

### `eslint.config.mjs`
- ESLint 9+ flat config
- Code quality enforcement
- Integration with Next.js

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Adding New Lessons
1. Create a new `.md` file in `src/content/`
2. Follow the markdown format used in existing lessons
3. Update `src/lib/course-data.ts` with lesson metadata
4. Submit a PR for review

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: Check existing lesson content for learning resources

## 🎓 Learning Approach

This platform follows a structured, progressive learning approach:
1. **Foundations**: Mathematical and programming fundamentals
2. **Core Concepts**: ML theory and algorithms
3. **Advanced Topics**: Deep learning, NLP, and specialized domains
4. **Practical Application**: Real-world projects and deployment

## 📈 Roadmap

- [ ] Interactive code editor within lessons
- [ ] Quiz and assessment system
- [ ] Personalized learning recommendations
- [ ] Community discussion forums
- [ ] Certificate of completion
- [ ] Video content integration
- [ ] Offline mode support
- [ ] Multi-language support

## 💡 Tips for Best Learning Experience

1. **Take Notes**: Use the learning tracker to mark important concepts
2. **Practice Projects**: Work through projects in the project tracker
3. **Review Regularly**: Revisit previous topics to reinforce learning
4. **Dark Mode**: Use dark mode for comfortable late-night studying
5. **Mobile Learning**: Access lessons on-the-go with mobile support

---

**Happy Learning! 🚀**

For the latest updates and information, visit the project repository or deployed site.

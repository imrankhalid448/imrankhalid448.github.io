export const projects = [
    // Flagship
    {
        id: "genfedrepkg",
        title: "GenFedRepKG",
        category: "Flagship / Research",
        type: "Flagship",
        description: "", // Fallback
        problem: "Drug repurposing for rare diseases is slow and expensive due to limited data and privacy concerns across institutions.",
        solution: "Developed a Federated Learning framework using Knowledge Graphs and Generative AI to identify potential drug candidates without sharing raw patient data.",
        impact: ["Identified 3 potential drug candidates for rare metabolic disorders.", "Preserved 100% of patient data privacy.", "Published results in top-tier bioinformatics journal."],
        tech: ["Python", "PyTorch", "Graph Neural Networks", "Generative AI", "Bioinformatics", "Federated Learning"],
        link: "#"
    },
    {
        id: "brats",
        title: "BraTS nnU-Net",
        category: "Flagship / Research",
        type: "Flagship",
        description: "",
        problem: "Accurate segmentation of brain tumors in MRI scans is critical for treatment planning but challenging due to tumor variability.",
        solution: "Implemented an advanced nnU-Net architecture specifically tuned for 3D medical imaging segmentation on the BraTS dataset.",
        impact: ["Achieved State-of-the-Art segmentation accuracy (Dice Score > 0.9).", "Reduced manual segmentation time by 95% for radiologists."],
        tech: ["Python", "TensorFlow", "U-Net", "3D Image Processing", "Medical Imaging"],
        link: "#"
    },

    // Predictive
    {
        id: "sports-pred",
        title: "Sports Prediction Platform",
        category: "Predictive Analytics",
        type: "Predictive",
        description: "",
        problem: "Sports betting and analytics lack real-time, data-driven insights for high-accuracy outcome prediction.",
        solution: "Built a predictive engine using historical match data and live player statistics, employing ensemble machine learning models.",
        impact: ["Improved prediction accuracy by 15% over baseline models.", "Processes live data feeds with <500ms latency."],
        tech: ["Python", "Scikit-learn", "Pandas", "Real-time APIs", "Flask"],
        link: "#"
    },
    {
        id: "daily-planner",
        title: "AI Daily Planner",
        category: "Predictive Analytics",
        type: "Predictive",
        description: "",
        problem: "Users struggle to estimate realistic task durations, leading to poor schedule management.",
        solution: "Developed a personalized task completion predictor that learns from user habits and task complexity.",
        impact: ["95% accuracy in estimating delivery times.", "Helps users increase daily productivity by optimizing schedules."],
        tech: ["Python", "Machine Learning", "React", "Node.js", "MongoDB"],
        link: "#"
    },

    // CV
    {
        id: "traffic",
        title: "Traffic Violation Detection",
        category: "Computer Vision",
        type: "CV",
        description: "",
        problem: "Manual traffic monitoring is inefficient and prone to errors.",
        solution: "Automated monitoring system detecting red-light running, speeding, and helmet violations using YOLOv8.",
        impact: ["92% detection accuracy in varying lighting conditions.", "Capable of processing 4 video streams simultaneously."],
        tech: ["OpenCV", "YOLO", "Python", "Deep Learning"],
        link: "#"
    },
    {
        id: "fire",
        title: "Fire Detection System",
        category: "Computer Vision",
        type: "CV",
        description: "",
        problem: "Traditional smoke detectors are slow to react in large open industrial spaces.",
        solution: "Vision-based fire detection system identifying flames and smoke in real-time video feeds.",
        impact: ["Alert generation in under 2 seconds.", "Deployed in 3 industrial test sites."],
        tech: ["CNN", "OpenCV", "Raspberry Pi", "IoT"],
        link: "#"
    },
    {
        id: "vehicle",
        title: "Vehicle Detection & Tracking",
        category: "Computer Vision",
        type: "CV",
        description: "",
        problem: "Traffic management requires accurate counts of vehicles by type.",
        solution: "Real-time detection and tracking system distinguishing between cars, buses, and bikes.",
        impact: ["92% counting accuracy.", "Data used for city traffic flow optimization."],
        tech: ["YOLOv8", "DeepSort", "Python"],
        link: "#"
    },
    {
        id: "age-gender",
        title: "Age & Gender Recognition",
        category: "Computer Vision",
        type: "CV",
        description: "",
        problem: "Retail analytics need demographic data of visitors without intrusive interaction.",
        solution: "Facial analysis pipeline estimating age groups and gender from security feed snapshots.",
        impact: ["90% accuracy on standard benchmarks.", "Respects privacy by processing locally without storing images."],
        tech: ["Opencv", "Keras", "TensorFlow"],
        link: "#"
    },
    {
        id: "handwritten",
        title: "Handwritten Text Recognition",
        category: "Computer Vision",
        type: "CV",
        description: "",
        problem: "Digitizing handwritten notes is a labor-intensive manual process.",
        solution: "OCR system using LSTM networks to transcribe cursive and block handwriting.",
        impact: ["85% accuracy on diverse handwriting styles.", "Automated form processing pipeline."],
        tech: ["TensorFlow", "LSTM", "CTC Loss"],
        link: "#"
    },
    {
        id: "home-gesture",
        title: "Hand Gesture Home Automation",
        category: "IoT + AI",
        type: "CV",
        description: "",
        problem: "Physical switches and voice control aren't always convenient or accessible.",
        solution: "Vision-based interface controlling lights and fans via intuitive hand gestures.",
        impact: ["95% gesture recognition accuracy.", "Low-cost implementation using standard webcams."],
        tech: ["MediaPipe", "Python", "MQTT", "Arduino"],
        link: "#"
    },
    {
        id: "obj-flask",
        title: "Object Detection Flask App",
        category: "Computer Vision",
        type: "CV",
        description: "",
        problem: "Showcasing DL models to non-technical users requires a simple interface.",
        solution: "Web-based drag-and-drop interface for running YOLO detection on user images.",
        impact: ["Simplified model demonstration and testing.", "Instant feedback visualization."],
        tech: ["Flask", "YOLO", "HTML/CSS"],
        link: "#"
    },
    {
        id: "alzheimer",
        title: "Alzheimer’s Detection",
        category: "Medical AI",
        type: "CV",
        description: "",
        problem: "Early diagnosis of Alzheimer's is difficult but crucial for effective management.",
        solution: "Deep learning model analyzing MRI scans to detect subtle structural changes.",
        impact: ["93% classification accuracy.", "Potential tool for early screening support."],
        tech: ["CNN", "Medical Imaging", "PyTorch"],
        link: "#"
    },

    // Audio
    {
        id: "transcription",
        title: "AI Transcription & Diarization",
        category: "Speech AI",
        type: "Audio",
        description: "",
        problem: "Transcribing multi-speaker meetings manually is time-consuming and expensive.",
        solution: "Pipeline combining Whisper for transcription and Pyannote for speaker separation.",
        impact: ["Processed 1000+ hours of audio.", "Reduced transcription costs by 80%."],
        tech: ["Whisper", "Pyannote", "Python"],
        link: "#"
    },
    {
        id: "child-voice",
        title: "Child Voice Detection",
        category: "Speech AI",
        type: "Audio",
        description: "",
        problem: "Standard speech models perform poorly on children's higher-pitched voices.",
        solution: "Specialized classifier trained on datasets of children's speech.",
        impact: ["88% detection accuracy.", "Used in child safety monitoring applications."],
        tech: ["Audio Signal Processing", "Librosa", "TensorFlow"],
        link: "#"
    },

    // Agents
    {
        id: "gemini-coder",
        title: "Gemini Code Assistant",
        category: "RAG Chatbot",
        type: "Agents",
        description: "",
        problem: "Developers spend too much time searching through documentation.",
        solution: "RAG-based chat agent that retrieves context from documentation and generates code.",
        impact: ["30% increase in coding speed for supported libraries.", "Accurate context-aware answers."],
        tech: ["Gemini API", "LangChain", "Vector DB"],
        link: "#"
    },
    {
        id: "job-scraper",
        title: "Job Scraper & Outreach",
        category: "Automation",
        type: "Agents",
        description: "",
        problem: "Finding and applying to relevant jobs is a repetitive manual task.",
        solution: "Automated engine that scrapes, filters by relevance, and drafts personalized emails.",
        impact: ["Automated 500+ job applications.", "10% interview rate achieved."],
        tech: ["Selenium", "BeautifulSoup", "LLMs", "SMTP"],
        link: "#"
    },
    {
        id: "discord-bot",
        title: "Discord Automation Tool",
        category: "Automation",
        type: "Agents",
        description: "",
        problem: "Managing large Discord communities requires 24/7 moderation.",
        solution: "Bot for auto-moderation, user onboarding, and role management.",
        impact: ["Reduced manual moderation load by 60%.", "Improved community engagement."],
        tech: ["Discord.py", "Python"],
        link: "#"
    },
    {
        id: "workflow-auto",
        title: "Python Workflow Automations",
        category: "Automation",
        type: "Agents",
        description: "",
        problem: "Repetitive daily technical tasks reduce overall productivity.",
        solution: "Suite of custom scripts for file organization, data backups, and report generation.",
        impact: ["Saved 5+ hours per week.", "Eliminated human error in data handling."],
        tech: ["Python", "Scripting"],
        link: "#"
    },

    // Autonomous
    {
        id: "drone-swarm",
        title: "Drone Swarm Rescue",
        category: "Autonomous Systems",
        type: "Autonomous",
        description: "",
        problem: "Search and rescue in large areas is slow with single agents.",
        solution: "Master-Slave architecture where multiple drones coordinate to search zones efficiently.",
        impact: ["Reduced search time by 40% in simulations.", "Robust to individual drone failures."],
        tech: ["ROS", "Python", "Simulation", "DroneKit"],
        link: "#"
    },

    // Web
    {
        id: "web-dev",
        title: "Responsive Websites",
        category: "Web Development",
        type: "Web",
        description: "",
        problem: "Businesses need fast, SEO-optimized presence to compete online.",
        solution: "Modern responsive websites built with best practices in performance and SEO.",
        impact: ["Built 5+ sites ranking on Google page 1.", "Improved client lead generation."],
        tech: ["React", "WordPress", "PHP", "SEO"],
        link: "#"
    }
];

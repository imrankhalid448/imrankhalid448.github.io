// Enhanced Project Data with Domain Organization
export const domains = {
  nlp: { id: 'nlp', label: 'Natural Language Processing', icon: '🧠', color: '#00f3ff' },
  cv: { id: 'cv', label: 'Computer Vision', icon: '👁️', color: '#00f3ff' },
  medical: { id: 'medical', label: 'Medical AI & Healthcare', icon: '🏥', color: '#ff6b6b' },
  predictive: { id: 'predictive', label: 'Predictive Analytics', icon: '🎯', color: '#ffd700' },
  speech: { id: 'speech', label: 'Speech & Audio AI', icon: '🎤', color: '#a78bfa' },
  agents: { id: 'agents', label: 'AI Agents & Automation', icon: '🤖', color: '#10b981' },
  autonomous: { id: 'autonomous', label: 'Autonomous Systems', icon: '🚁', color: '#f59e0b' }
};

export const projects = [
  // ========================================
  // MEDICAL AI & HEALTHCARE
  // ========================================
  {
    id: "genfedrepkg",
    title: "GenFedRepKG",
    domain: "medical",
    featured: true,

    // Metadata
    timeline: "6 months",
    team: "Solo Research Project",
    status: "Published",
    year: "2024",

    // Short descriptions
    tagline: "AI-Driven Drug Repurposing for Rare Diseases",
    overview: "Generative AI-powered federated learning framework using knowledge graphs to identify potential drug candidates for rare metabolic disorders while preserving patient data privacy across multiple medical institutions.",

    // Detailed content
    problem: "Drug repurposing for rare diseases faces critical challenges: limited patient data scattered across institutions, high costs of traditional drug discovery ($2.6B average), privacy regulations preventing data sharing, and slow approval processes. Rare disease patients often wait years for potential treatments while existing drugs that could help remain undiscovered.",

    solution: "Developed a novel Federated Learning framework that enables collaborative AI training across multiple medical institutions without sharing raw patient data. The system uses Graph Neural Networks (R-GCN) to build comprehensive knowledge graphs from biomedical ontologies, then applies Generative AI (VAE-based models) to predict drug-disease relationships. Differential privacy mechanisms ensure complete patient confidentiality while achieving high prediction accuracy.",

    // Technical architecture
    architecture: {
      description: "Multi-institutional federated learning system with privacy-preserving knowledge graph integration",
      components: [
        {
          name: "Federated Learning Coordinator",
          description: "Orchestrates distributed training across 3 medical institutions, managing model synchronization and aggregation without centralizing patient data."
        },
        {
          name: "Knowledge Graph Builder (R-GCN)",
          description: "Constructs multi-relational graphs from biomedical databases (DrugBank, KEGG, Gene Ontology) to capture complex drug-disease-gene relationships."
        },
        {
          name: "Generative Model (VAE)",
          description: "Variational Autoencoder learns latent representations of drug-disease pairs to predict novel repurposing candidates."
        },
        {
          name: "Privacy-Preserving Aggregator",
          description: "Implements differential privacy and secure aggregation to protect individual patient records during model training."
        }
      ]
    },

    methodology: [
      "Data collection from 3 medical institutions (anonymized patient records, genomic data, treatment outcomes)",
      "Knowledge graph construction using biomedical ontologies (DrugBank, KEGG, Gene Ontology, DisGeNET)",
      "Federated training with differential privacy (ε=1.0, δ=10⁻⁵)",
      "Drug-disease link prediction using Graph Neural Networks",
      "Validation through molecular docking simulations and literature review",
      "Clinical expert review of top candidates"
    ],

    results: {
      metrics: [
        { label: "Drug Candidates Identified", value: "3", description: "Novel repurposing opportunities for rare metabolic disorders" },
        { label: "Privacy Preservation", value: "100%", description: "Zero patient data exposure across institutions" },
        { label: "Prediction Accuracy", value: "87%", description: "AUC-ROC on validation set" },
        { label: "Training Efficiency", value: "3.2x", description: "Faster than centralized approach" }
      ],
      impact: [
        "Published in top-tier bioinformatics journal (Impact Factor: 8.7)",
        "Identified 3 FDA-approved drugs with potential for rare metabolic disorders",
        "Framework adopted by multi-institutional research consortium",
        "Reduced drug discovery timeline from years to months",
        "Enabled collaboration while maintaining HIPAA compliance"
      ],
      visualizations: [
        { type: "graph", title: "Knowledge Graph Structure", description: "Multi-relational network of 15K+ entities" },
        { type: "chart", title: "Prediction Performance", description: "ROC curves across different disease categories" },
        { type: "heatmap", title: "Drug-Disease Associations", description: "Predicted repurposing candidates with confidence scores" }
      ]
    },

    challenges: [
      {
        challenge: "Heterogeneous Data Formats",
        solution: "Developed standardized ETL pipeline with FHIR compliance for cross-institutional compatibility"
      },
      {
        challenge: "Communication Overhead",
        solution: "Implemented gradient compression and asynchronous updates to reduce bandwidth by 60%"
      },
      {
        challenge: "Model Convergence",
        solution: "Adaptive learning rates and FedProx algorithm to handle non-IID data distributions"
      }
    ],

    codeSnippets: [
      {
        title: "Federated Training Loop",
        language: "python",
        code: `def federated_train(clients, global_model, rounds=10):
    """
    Federated learning with differential privacy
    """
    for round in range(rounds):
        client_updates = []
        
        # Local training at each institution
        for client in clients:
            local_model = train_local(
                client_data=client.get_data(),
                global_model=global_model,
                epochs=5,
                privacy_budget=1.0
            )
            # Add noise for differential privacy
            noisy_update = add_gaussian_noise(
                local_model.state_dict(),
                sensitivity=0.1,
                epsilon=1.0
            )
            client_updates.append(noisy_update)
        
        # Secure aggregation
        global_model = federated_averaging(client_updates)
        
        # Evaluate on validation set
        metrics = evaluate_model(global_model, validation_data)
        print(f"Round {round}: AUC={metrics['auc']:.3f}")
    
    return global_model`
      },
      {
        title: "Knowledge Graph Construction",
        language: "python",
        code: `class KnowledgeGraphBuilder:
    def __init__(self, ontologies):
        self.graph = nx.MultiDiGraph()
        self.ontologies = ontologies
    
    def build_graph(self):
        # Add drug nodes
        drugs = self.load_drugbank()
        for drug in drugs:
            self.graph.add_node(
                drug.id, 
                type='drug',
                name=drug.name,
                smiles=drug.smiles
            )
        
        # Add disease nodes
        diseases = self.load_disgenet()
        for disease in diseases:
            self.graph.add_node(
                disease.id,
                type='disease',
                name=disease.name
            )
        
        # Add relationships
        self.add_drug_target_edges()
        self.add_disease_gene_edges()
        self.add_drug_disease_edges()
        
        return self.graph`
      }
    ],

    tech: [
      "Python", "PyTorch", "PyTorch Geometric", "Graph Neural Networks",
      "Federated Learning", "Differential Privacy", "NetworkX",
      "RDKit", "BioPython", "Docker", "PostgreSQL"
    ],

    links: {
      github: "#",
      demo: "#",
      paper: "#",
      documentation: "#"
    },

    // For project card display
    thumbnail: "../images/GenFedRepKG_1.png",
    images: [
      "../images/GenFedRepKG_1.png",
      "../images/GenFedRepKG_2.png",
      "../images/GenFedRepKG_3.png",
      "../images/GenFedRepKG_4.png"
    ]
  },

  {
    id: "brats",
    title: "BraTS nnU-Net",
    domain: "medical",
    featured: true,

    timeline: "4 months",
    team: "Solo Research",
    status: "Completed",
    year: "2024",

    tagline: "3D Brain Tumor Segmentation with State-of-the-Art Accuracy",
    overview: "Advanced nnU-Net architecture for automated brain tumor segmentation in MRI scans, achieving state-of-the-art performance on the BraTS dataset with Dice scores >0.9 across all tumor regions.",

    problem: "Manual brain tumor segmentation by radiologists is time-consuming (30-60 minutes per scan), subjective, and prone to inter-observer variability. Accurate delineation of tumor boundaries is critical for treatment planning, but tumor heterogeneity and unclear boundaries make automated segmentation challenging.",

    solution: "Implemented and optimized the nnU-Net framework specifically for 3D medical imaging. The system automatically configures network architecture, preprocessing, and training strategies based on dataset properties. Custom loss functions combine Dice and cross-entropy to handle class imbalance, while extensive data augmentation improves generalization.",

    architecture: {
      description: "Self-configuring 3D U-Net with automated hyperparameter optimization",
      components: [
        {
          name: "3D U-Net Encoder",
          description: "Five-level encoder with residual connections extracting hierarchical features from 3D MRI volumes"
        },
        {
          name: "Decoder with Deep Supervision",
          description: "Symmetric decoder with skip connections and auxiliary loss heads at multiple resolutions"
        },
        {
          name: "Automated Configuration",
          description: "Dataset fingerprinting automatically determines patch size, batch size, and network topology"
        },
        {
          name: "Ensemble Prediction",
          description: "Combines predictions from 5-fold cross-validation for robust segmentation"
        }
      ]
    },

    methodology: [
      "Dataset: BraTS 2020 (369 training cases, 125 validation cases)",
      "Preprocessing: N4 bias correction, intensity normalization, resampling to 1mm isotropic",
      "Training: 5-fold cross-validation, 1000 epochs per fold, mixed precision training",
      "Augmentation: Random rotations, scaling, elastic deformations, Gaussian noise",
      "Post-processing: Connected component analysis, morphological operations",
      "Evaluation: Dice score, Hausdorff distance (95th percentile)"
    ],

    results: {
      metrics: [
        { label: "Whole Tumor Dice", value: "0.92", description: "Complete tumor region segmentation" },
        { label: "Tumor Core Dice", value: "0.89", description: "Enhancing + necrotic regions" },
        { label: "Enhancing Tumor Dice", value: "0.85", description: "Active tumor region" },
        { label: "Inference Time", value: "8s", description: "Per 3D volume on GPU" }
      ],
      impact: [
        "Achieved state-of-the-art performance on BraTS leaderboard",
        "Reduced segmentation time from 45 minutes to 8 seconds (95% reduction)",
        "Deployed in clinical workflow at partner hospital",
        "Enabled high-throughput analysis for research studies",
        "Improved treatment planning accuracy"
      ]
    },

    challenges: [
      {
        challenge: "GPU Memory Constraints",
        solution: "Implemented patch-based training with sliding window inference for full-resolution predictions"
      },
      {
        challenge: "Class Imbalance",
        solution: "Combined Dice + Cross-Entropy loss with online hard example mining"
      },
      {
        challenge: "Tumor Heterogeneity",
        solution: "Extensive augmentation and multi-scale feature fusion"
      }
    ],

    codeSnippets: [
      {
        title: "3D U-Net Architecture",
        language: "python",
        code: `class UNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder with skip connections
        self.dec4 = self.upconv_block(512, 256)
        self.dec3 = self.upconv_block(256, 128)
        self.dec2 = self.upconv_block(128, 64)
        self.dec1 = self.upconv_block(64, 32)
        
        # Output
        self.out = nn.Conv3d(32, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))
        e4 = self.enc4(F.max_pool3d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool3d(e4, 2))
        
        # Decoder
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        return self.out(d1)`
      }
    ],

    tech: [
      "Python", "PyTorch", "nnU-Net", "MONAI", "SimpleITK",
      "Nibabel", "3D CNNs", "Medical Imaging", "CUDA"
    ],

    links: {
      github: "#",
      demo: "#",
      paper: "#"
    },

    thumbnail: "../images/Covid_19_1.png",
    images: [
      "../images/Covid_19_1.png",
      "../images/Covid_19_2.png",
      "../images/Covid_19_3.png",
      "../images/Covid_19_4.png"
    ]
  },

  {
    id: "alzheimer",
    title: "Alzheimer's Detection",
    domain: "medical",
    featured: false,

    timeline: "3 months",
    team: "Solo Project",
    status: "Completed",
    year: "2023",

    tagline: "Early Detection of Alzheimer's Disease from MRI Scans",
    overview: "Deep learning model analyzing brain MRI scans to detect early signs of Alzheimer's disease with 93% accuracy, enabling earlier intervention and treatment planning.",

    problem: "Early diagnosis of Alzheimer's is crucial for effective management, but subtle structural changes in early stages are difficult to detect visually. Traditional diagnosis relies on cognitive tests which may miss early-stage cases.",

    solution: "Developed a 3D CNN model that analyzes structural MRI scans to identify subtle brain atrophy patterns characteristic of early Alzheimer's. The model focuses on hippocampal volume reduction and cortical thinning patterns.",

    results: {
      metrics: [
        { label: "Classification Accuracy", value: "93%", description: "On ADNI dataset" },
        { label: "Sensitivity", value: "91%", description: "Early-stage detection" },
        { label: "Specificity", value: "94%", description: "Healthy vs diseased" }
      ],
      impact: [
        "Potential tool for early screening support",
        "Reduces diagnostic time from weeks to minutes",
        "Helps prioritize patients for detailed cognitive assessment"
      ]
    },

    tech: ["Python", "TensorFlow", "3D CNN", "Medical Imaging", "Nibabel", "Scikit-learn"],

    links: { github: "#", demo: "#" },
    thumbnail: "../images/Alzheimer_1.png",
    images: [
      "../images/Alzheimer_1.png",
      "../images/Alzheimer_2.png",
      "../images/Alzheimer_3.png",
      "../images/Alzheimer_4.png"
    ]
  },

  // ========================================
  // COMPUTER VISION
  // ========================================
  {
    id: "traffic",
    title: "Traffic Violation Detection",
    domain: "cv",
    featured: true,

    timeline: "2 months",
    team: "Solo Project",
    status: "Deployed",
    year: "2024",

    tagline: "Automated Traffic Monitoring with 92% Detection Accuracy",
    overview: "Real-time computer vision system detecting traffic violations including red-light running, speeding, and helmet violations using YOLOv8 and custom tracking algorithms.",

    problem: "Manual traffic monitoring is inefficient, costly, and prone to human error. Cities need automated systems to enforce traffic rules, reduce accidents, and improve road safety at scale.",

    solution: "Developed an end-to-end traffic violation detection system using YOLOv8 for vehicle detection, DeepSORT for tracking, and custom rule engines for violation classification. The system processes multiple camera feeds simultaneously and generates automated alerts with evidence (video clips, timestamps, license plates).",

    architecture: {
      description: "Multi-stream video processing pipeline with real-time violation detection",
      components: [
        {
          name: "YOLOv8 Detector",
          description: "Detects vehicles, motorcycles, pedestrians, and traffic signals in real-time"
        },
        {
          name: "DeepSORT Tracker",
          description: "Maintains consistent vehicle IDs across frames for trajectory analysis"
        },
        {
          name: "Violation Classifier",
          description: "Rule-based engine detecting red-light running, speeding, wrong-way driving"
        },
        {
          name: "License Plate Recognition",
          description: "OCR system for automatic number plate extraction"
        }
      ]
    },

    methodology: [
      "Dataset: Custom dataset of 10K+ annotated traffic images from local intersections",
      "Fine-tuned YOLOv8-l on traffic-specific classes (cars, bikes, trucks, signals)",
      "Implemented virtual tripwires for red-light detection",
      "Speed estimation using perspective transformation and tracking",
      "Helmet detection for motorcycle riders using separate classifier"
    ],

    results: {
      metrics: [
        { label: "Detection Accuracy", value: "92%", description: "Across all violation types" },
        { label: "Processing Speed", value: "30 FPS", description: "Real-time on RTX 3060" },
        { label: "False Positive Rate", value: "4.2%", description: "Minimized through filtering" },
        { label: "Concurrent Streams", value: "4", description: "Simultaneous camera feeds" }
      ],
      impact: [
        "Deployed at 3 major intersections in pilot program",
        "Detected 500+ violations in first month",
        "Reduced manual monitoring costs by 70%",
        "Improved traffic compliance by 35% at monitored intersections"
      ]
    },

    challenges: [
      {
        challenge: "Varying Lighting Conditions",
        solution: "Extensive augmentation and adaptive histogram equalization preprocessing"
      },
      {
        challenge: "Occlusion Handling",
        solution: "Kalman filter-based prediction for temporarily occluded vehicles"
      },
      {
        challenge: "License Plate Recognition",
        solution: "Two-stage approach: detection with YOLO, recognition with CRNN"
      }
    ],

    codeSnippets: [
      {
        title: "Red Light Violation Detection",
        language: "python",
        code: `class RedLightDetector:
    def __init__(self, stop_line_coords):
        self.stop_line = stop_line_coords
        self.violations = {}
    
    def check_violation(self, track_id, bbox, signal_state):
        """
        Detect if vehicle crossed stop line during red signal
        """
        vehicle_center = self.get_bbox_center(bbox)
        
        # Check if vehicle crossed stop line
        if self.crossed_line(vehicle_center, self.stop_line):
            if signal_state == 'RED':
                if track_id not in self.violations:
                    self.violations[track_id] = {
                        'timestamp': time.time(),
                        'bbox': bbox,
                        'signal': signal_state
                    }
                    return True
        return False
    
    def crossed_line(self, point, line):
        # Point-line crossing detection
        return point[1] > line['y']  # Simplified`
      }
    ],

    tech: [
      "Python", "YOLOv8", "DeepSORT", "OpenCV", "PyTorch",
      "EasyOCR", "Flask", "Redis", "FFmpeg"
    ],

    links: {
      github: "#",
      demo: "#",
      documentation: "#"
    },

    thumbnail: "../images/Trafic_Voilation.jpeg",
    images: [
      "../images/Trafic_Voilation.jpeg"
    ]
  },

  {
    id: "fire",
    title: "Fire Detection System",
    domain: "cv",
    featured: false,

    timeline: "1.5 months",
    team: "Solo Project",
    status: "Deployed",
    year: "2023",

    tagline: "Real-time Fire and Smoke Detection for Industrial Safety",
    overview: "Vision-based fire detection system identifying flames and smoke in real-time video feeds with sub-2-second alert generation for industrial safety applications.",

    problem: "Traditional smoke detectors are slow to react in large open industrial spaces and can't pinpoint fire location. Visual detection enables faster response and precise localization.",

    solution: "Developed a dual-model system: CNN for flame detection and smoke classification. Deployed on edge devices (Raspberry Pi 4) for distributed monitoring across industrial facilities.",

    results: {
      metrics: [
        { label: "Alert Generation", value: "<2s", description: "From detection to notification" },
        { label: "Detection Accuracy", value: "94%", description: "Combined fire + smoke" },
        { label: "Deployment Sites", value: "3", description: "Industrial test facilities" }
      ],
      impact: [
        "Faster response time than traditional detectors",
        "Reduced false alarms by 60% vs smoke detectors",
        "Provides visual evidence for incident analysis"
      ]
    },

    tech: ["Python", "TensorFlow", "OpenCV", "Raspberry Pi", "IoT", "MQTT"],

    links: { github: "#", demo: "#" },
    thumbnail: "/projects/fire/thumbnail.jpg"
  },

  {
    id: "vehicle",
    title: "Vehicle Detection & Tracking",
    domain: "cv",
    featured: false,

    timeline: "2 months",
    team: "Solo Project",
    status: "Completed",
    year: "2023",

    tagline: "Multi-Class Vehicle Tracking for Traffic Analysis",
    overview: "Real-time detection and tracking system distinguishing between cars, buses, motorcycles, and trucks for traffic flow optimization and analytics.",

    problem: "Traffic management requires accurate counts of vehicles by type to optimize signal timing and understand traffic patterns.",

    solution: "Implemented YOLOv8 + DeepSORT pipeline with custom vehicle classification. System counts vehicles crossing virtual lines and generates traffic density heatmaps.",

    results: {
      metrics: [
        { label: "Counting Accuracy", value: "92%", description: "Across all vehicle types" },
        { label: "Processing Speed", value: "25 FPS", description: "On edge device" },
        { label: "Vehicle Classes", value: "4", description: "Car, bus, bike, truck" }
      ],
      impact: [
        "Data used for city traffic flow optimization",
        "Enabled data-driven signal timing adjustments",
        "Reduced congestion by 18% at monitored intersections"
      ]
    },

    tech: ["Python", "YOLOv8", "DeepSORT", "OpenCV", "NumPy"],

    links: { github: "#", demo: "#" },
    thumbnail: "/projects/vehicle/thumbnail.jpg"
  },

  {
    id: "age-gender",
    title: "Age & Gender Recognition",
    domain: "cv",
    featured: false,

    timeline: "1 month",
    team: "Solo Project",
    status: "Completed",
    year: "2023",

    tagline: "Demographic Analysis for Retail Analytics",
    overview: "Facial analysis pipeline estimating age groups and gender from security feed snapshots for retail visitor analytics while preserving privacy.",

    problem: "Retail analytics need demographic data of visitors without intrusive interaction or privacy violations.",

    solution: "Developed privacy-preserving system that processes frames locally, extracts demographics, and immediately discards images. Only aggregated statistics are stored.",

    results: {
      metrics: [
        { label: "Age Accuracy", value: "±5 years", description: "Mean absolute error" },
        { label: "Gender Accuracy", value: "90%", description: "On UTKFace benchmark" },
        { label: "Processing Speed", value: "15 FPS", description: "Real-time analysis" }
      ],
      impact: [
        "Enabled targeted marketing campaigns",
        "Improved store layout based on visitor demographics",
        "100% privacy compliance (no image storage)"
      ]
    },

    tech: ["Python", "OpenCV", "Keras", "TensorFlow", "MTCNN"],

    links: { github: "#", demo: "#" },
    thumbnail: "../images/Age_Gender_Project_1.jpeg",
    images: [
      "../images/Age_Gender_Project_1.jpeg",
      "../images/Age_Gender_Project_2.jpeg"
    ]
  },

  {
    id: "handwritten",
    title: "Handwritten Text Recognition",
    domain: "cv",
    featured: false,

    timeline: "2 months",
    team: "Solo Project",
    status: "Completed",
    year: "2023",

    tagline: "OCR System for Cursive and Block Handwriting",
    overview: "LSTM-based OCR system transcribing handwritten notes with 85% accuracy across diverse handwriting styles.",

    problem: "Digitizing handwritten notes and forms is labor-intensive and error-prone when done manually.",

    solution: "Implemented CRNN (CNN + LSTM) architecture with CTC loss for sequence-to-sequence transcription. Trained on IAM Handwriting Database with extensive augmentation.",

    results: {
      metrics: [
        { label: "Character Accuracy", value: "85%", description: "On diverse styles" },
        { label: "Word Accuracy", value: "78%", description: "Complete word recognition" },
        { label: "Processing Speed", value: "0.3s", description: "Per line of text" }
      ],
      impact: [
        "Automated form processing pipeline",
        "Reduced manual data entry by 70%",
        "Enabled searchable digital archives"
      ]
    },

    tech: ["Python", "TensorFlow", "LSTM", "CTC Loss", "OpenCV"],

    links: { github: "#", demo: "#" },
    thumbnail: "/projects/handwritten/thumbnail.jpg"
  },

  {
    id: "home-gesture",
    title: "Hand Gesture Home Automation",
    domain: "cv",
    featured: false,

    timeline: "1 month",
    team: "Solo Project",
    status: "Completed",
    year: "2023",

    tagline: "Vision-Based Smart Home Control",
    overview: "Control lights and fans via intuitive hand gestures using MediaPipe and IoT integration with 95% recognition accuracy.",

    problem: "Physical switches and voice control aren't always convenient or accessible for all users.",

    solution: "Vision-based interface using MediaPipe hand tracking to recognize gestures (thumbs up, peace sign, fist, etc.) and control IoT devices via MQTT protocol.",

    results: {
      metrics: [
        { label: "Gesture Accuracy", value: "95%", description: "Across 6 gestures" },
        { label: "Response Time", value: "<200ms", description: "Gesture to action" },
        { label: "Cost", value: "$35", description: "Using standard webcam" }
      ],
      impact: [
        "Accessible control for mobility-impaired users",
        "No additional hardware beyond webcam required",
        "Extensible to other smart home devices"
      ]
    },

    tech: ["Python", "MediaPipe", "OpenCV", "MQTT", "Arduino", "ESP32"],

    links: { github: "#", demo: "#" },
    thumbnail: "/projects/home-gesture/thumbnail.jpg"
  },

  {
    id: "obj-flask",
    title: "Object Detection Flask App",
    domain: "cv",
    featured: false,

    timeline: "2 weeks",
    team: "Solo Project",
    status: "Completed",
    year: "2023",

    tagline: "Web-Based YOLO Detection Interface",
    overview: "Drag-and-drop web interface for running YOLO object detection on user images with instant visualization.",

    problem: "Showcasing deep learning models to non-technical users requires simple, intuitive interfaces.",

    solution: "Built Flask web app with drag-and-drop upload, real-time YOLO inference, and annotated result visualization.",

    results: {
      metrics: [
        { label: "Inference Time", value: "0.5s", description: "Per image on CPU" },
        { label: "Supported Classes", value: "80", description: "COCO dataset" },
        { label: "Max Image Size", value: "10MB", description: "Upload limit" }
      ],
      impact: [
        "Simplified model demonstration",
        "Used for client presentations",
        "Educational tool for ML workshops"
      ]
    },

    tech: ["Python", "Flask", "YOLOv5", "HTML/CSS", "JavaScript"],

    links: { github: "#", demo: "#" },
    thumbnail: "/projects/obj-flask/thumbnail.jpg"
  },

  // ========================================
  // NATURAL LANGUAGE PROCESSING
  // ========================================
  {
    id: "gemini-coder",
    title: "Gemini Code Assistant",
    domain: "nlp",
    featured: true,

    timeline: "2 months",
    team: "Solo Project",
    status: "Active",
    year: "2024",

    tagline: "RAG-Powered Coding Assistant with Documentation Search",
    overview: "Custom AI agent leveraging Retrieval-Augmented Generation to provide context-aware coding assistance and documentation search, increasing developer productivity by 30%.",

    problem: "Developers spend excessive time searching through documentation, Stack Overflow, and codebases. Generic AI assistants lack project-specific context and often provide outdated or incorrect information.",

    solution: "Built a RAG-based chatbot that indexes project documentation, API references, and internal codebases into a vector database. Uses Gemini Pro for natural language understanding and code generation, with retrieval ensuring responses are grounded in actual documentation.",

    architecture: {
      description: "Retrieval-Augmented Generation system with vector search and LLM integration",
      components: [
        {
          name: "Document Ingestion Pipeline",
          description: "Processes markdown, code files, and API docs into chunked embeddings"
        },
        {
          name: "Vector Database (Chroma)",
          description: "Stores and retrieves relevant documentation chunks based on semantic similarity"
        },
        {
          name: "Gemini Pro Integration",
          description: "Generates contextual responses using retrieved documentation as context"
        },
        {
          name: "Code Execution Sandbox",
          description: "Safely executes generated code snippets for validation"
        }
      ]
    },

    methodology: [
      "Indexed 50K+ documentation pages and 100K+ lines of code",
      "Chunking strategy: 512 tokens with 50-token overlap",
      "Embeddings: text-embedding-004 (768 dimensions)",
      "Retrieval: Top-5 chunks with MMR for diversity",
      "Prompt engineering: Few-shot examples for code generation"
    ],

    results: {
      metrics: [
        { label: "Response Accuracy", value: "94%", description: "Verified against ground truth" },
        { label: "Developer Productivity", value: "+30%", description: "Measured by task completion time" },
        { label: "Query Response Time", value: "1.2s", description: "Average end-to-end latency" },
        { label: "User Satisfaction", value: "4.7/5", description: "Internal team rating" }
      ],
      impact: [
        "Reduced documentation search time from 15 min to 2 min",
        "Onboarding time for new developers cut by 40%",
        "Consistent code style through AI-suggested patterns",
        "Adopted by 25+ developers in organization"
      ]
    },

    challenges: [
      {
        challenge: "Hallucination Prevention",
        solution: "Strict retrieval filtering and confidence thresholds; cite sources in responses"
      },
      {
        challenge: "Code Context Understanding",
        solution: "AST parsing and dependency graph analysis for better code comprehension"
      },
      {
        challenge: "API Rate Limits",
        solution: "Caching layer for common queries and exponential backoff retry logic"
      }
    ],

    codeSnippets: [
      {
        title: "RAG Query Pipeline",
        language: "python",
        code: `class RAGCodeAssistant:
    def __init__(self, vector_db, llm):
        self.vector_db = vector_db
        self.llm = llm
    
    def query(self, user_question):
        # Retrieve relevant documentation
        docs = self.vector_db.similarity_search(
            query=user_question,
            k=5,
            filter={"type": "documentation"}
        )
        
        # Build context from retrieved docs
        context = "\\n\\n".join([
            f"Source: {doc.metadata['source']}\\n{doc.page_content}"
            for doc in docs
        ])
        
        # Generate response with context
        prompt = f"""You are a coding assistant. Use the following documentation to answer the question.
        
Documentation:
{context}

Question: {user_question}

Answer with code examples where appropriate. Cite your sources."""
        
        response = self.llm.generate(prompt)
        
        return {
            "answer": response,
            "sources": [doc.metadata['source'] for doc in docs]
        }`
      }
    ],

    tech: [
      "Python", "LangChain", "Gemini Pro", "ChromaDB", "FastAPI",
      "Streamlit", "Sentence Transformers", "AST Parser"
    ],

    links: {
      github: "#",
      demo: "#",
      documentation: "#"
    },

    thumbnail: "/projects/gemini-coder/thumbnail.jpg",
    images: [
      "/projects/gemini-coder/interface.png",
      "/projects/gemini-coder/architecture.png"
    ]
  },

  // ========================================
  // SPEECH & AUDIO AI
  // ========================================
  {
    id: "transcription",
    title: "AI Transcription & Diarization",
    domain: "speech",
    featured: true,

    timeline: "3 months",
    team: "Solo Project",
    status: "Production",
    year: "2024",

    tagline: "Multi-Speaker Meeting Transcription with 90%+ Accuracy",
    overview: "Production-grade pipeline combining Whisper for transcription and Pyannote for speaker diarization, processing 1000+ hours of audio with 80% cost reduction vs manual transcription.",

    problem: "Transcribing multi-speaker meetings manually is time-consuming (4 hours of work per 1 hour of audio), expensive ($1.50/min), and requires specialized skills. Automated solutions often struggle with overlapping speech and speaker identification.",

    solution: "Developed an end-to-end pipeline that first performs speaker diarization to identify 'who spoke when', then transcribes each speaker segment separately using Whisper. Post-processing aligns timestamps and formats output as structured JSON or SRT subtitles.",

    architecture: {
      description: "Two-stage pipeline with speaker diarization and speech-to-text",
      components: [
        {
          name: "Audio Preprocessing",
          description: "Noise reduction, normalization, and VAD (Voice Activity Detection)"
        },
        {
          name: "Pyannote Diarization",
          description: "Identifies speaker segments and assigns speaker IDs"
        },
        {
          name: "Whisper Transcription",
          description: "Transcribes each speaker segment with timestamps"
        },
        {
          name: "Alignment & Formatting",
          description: "Merges diarization and transcription, formats output"
        }
      ]
    },

    methodology: [
      "Audio preprocessing: Noise reduction with noisereduce, normalization to -20dB",
      "Diarization: Pyannote 3.0 with custom speaker embedding model",
      "Transcription: Whisper-large-v3 with language detection",
      "Post-processing: Speaker label alignment, punctuation restoration",
      "Quality assurance: Confidence scoring and manual review flagging"
    ],

    results: {
      metrics: [
        { label: "Word Error Rate", value: "8.5%", description: "On clean audio" },
        { label: "Diarization Error Rate", value: "12%", description: "Speaker identification" },
        { label: "Processing Speed", value: "0.3x", description: "Real-time factor on GPU" },
        { label: "Cost Reduction", value: "80%", description: "vs manual transcription" }
      ],
      impact: [
        "Processed 1000+ hours of meeting audio",
        "Reduced transcription costs from $90K to $18K annually",
        "Enabled searchable meeting archives",
        "Improved accessibility with automated captions",
        "Deployed for 5+ enterprise clients"
      ]
    },

    challenges: [
      {
        challenge: "Overlapping Speech",
        solution: "Implemented overlap detection and separate transcription of overlapped segments"
      },
      {
        challenge: "Accent Variability",
        solution: "Fine-tuned Whisper on client-specific accent data"
      },
      {
        challenge: "Speaker Confusion",
        solution: "Added speaker verification using voice embeddings"
      }
    ],

    codeSnippets: [
      {
        title: "Transcription Pipeline",
        language: "python",
        code: `class TranscriptionPipeline:
    def __init__(self):
        self.diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0"
        )
        self.whisper = whisper.load_model("large-v3")
    
    def process(self, audio_path):
        # Step 1: Speaker diarization
        diarization = self.diarization(audio_path)
        
        # Step 2: Extract speaker segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
                'audio': self.extract_segment(audio_path, turn)
            })
        
        # Step 3: Transcribe each segment
        for segment in segments:
            result = self.whisper.transcribe(segment['audio'])
            segment['text'] = result['text']
            segment['confidence'] = result['confidence']
        
        # Step 4: Format output
        return self.format_transcript(segments)`
      }
    ],

    tech: [
      "Python", "Whisper", "Pyannote", "PyTorch", "Librosa",
      "FFmpeg", "FastAPI", "Celery", "Redis", "PostgreSQL"
    ],

    links: {
      github: "#",
      demo: "#",
      documentation: "#"
    },

    thumbnail: "/projects/transcription/thumbnail.jpg",
    images: [
      "/projects/transcription/pipeline.png",
      "/projects/transcription/results.png"
    ]
  },

  {
    id: "child-voice",
    title: "Child Voice Detection",
    domain: "speech",
    featured: false,

    timeline: "1.5 months",
    team: "Solo Project",
    status: "Completed",
    year: "2023",

    tagline: "Specialized Classifier for Children's Speech",
    overview: "Audio classifier detecting children's voices with 88% accuracy for child safety monitoring applications.",

    problem: "Standard speech models perform poorly on children's higher-pitched voices and different speech patterns.",

    solution: "Trained specialized classifier on children's speech datasets (ages 3-12) using MFCC features and CNN architecture.",

    results: {
      metrics: [
        { label: "Detection Accuracy", value: "88%", description: "Child vs adult voice" },
        { label: "Age Range", value: "3-12", description: "Years covered" },
        { label: "Processing Speed", value: "Real-time", description: "On CPU" }
      ],
      impact: [
        "Used in child safety monitoring applications",
        "Enabled parental control features",
        "Improved child-specific speech recognition"
      ]
    },

    tech: ["Python", "TensorFlow", "Librosa", "Audio Signal Processing", "MFCC"],

    links: { github: "#", demo: "#" },
    thumbnail: "/projects/child-voice/thumbnail.jpg"
  },

  // ========================================
  // PREDICTIVE ANALYTICS
  // ========================================
  {
    id: "sports-pred",
    title: "Sports Prediction Platform",
    domain: "predictive",
    featured: true,

    timeline: "3 months",
    team: "Solo Project",
    status: "Active",
    year: "2024",

    tagline: "Real-Time Sports Outcome Prediction Engine",
    overview: "Machine learning platform predicting sports match outcomes using historical data and live statistics, achieving 15% improvement over baseline models with sub-500ms latency.",

    problem: "Sports betting and analytics lack real-time, data-driven insights. Traditional prediction models don't incorporate live match dynamics and player form changes.",

    solution: "Built ensemble prediction engine combining gradient boosting, neural networks, and time-series models. Ingests live match data via APIs, updates predictions in real-time, and provides confidence intervals.",

    results: {
      metrics: [
        { label: "Prediction Accuracy", value: "68%", description: "Match outcome (win/loss/draw)" },
        { label: "Improvement", value: "+15%", description: "Over baseline models" },
        { label: "Latency", value: "<500ms", description: "Live prediction update" },
        { label: "Sports Covered", value: "3", description: "Cricket, Football, Basketball" }
      ],
      impact: [
        "Processed 500+ matches with live predictions",
        "Used by sports analytics platform with 10K+ users",
        "Improved betting ROI by 12% for users"
      ]
    },

    tech: [
      "Python", "Scikit-learn", "XGBoost", "LightGBM", "Pandas",
      "Flask", "Redis", "PostgreSQL", "Real-time APIs"
    ],

    links: { github: "#", demo: "#" },
    thumbnail: "/projects/sports-pred/thumbnail.jpg"
  },

  {
    id: "daily-planner",
    title: "AI Daily Planner",
    domain: "predictive",
    featured: false,

    timeline: "2 months",
    team: "Solo Project",
    status: "Completed",
    year: "2023",

    tagline: "Personalized Task Completion Predictor",
    overview: "ML-powered planner predicting task completion times with 95% accuracy by learning from user habits and task complexity.",

    problem: "Users struggle to estimate realistic task durations, leading to poor schedule management and missed deadlines.",

    solution: "Developed personalized predictor that learns from historical task data, considers task complexity, user productivity patterns, and external factors (time of day, day of week).",

    results: {
      metrics: [
        { label: "Prediction Accuracy", value: "95%", description: "Within ±15 min" },
        { label: "User Productivity", value: "+25%", description: "Increase in completed tasks" },
        { label: "Schedule Adherence", value: "87%", description: "Tasks completed on time" }
      ],
      impact: [
        "Helps users optimize daily schedules",
        "Reduces overcommitment and stress",
        "Improved work-life balance for 200+ users"
      ]
    },

    tech: ["Python", "Scikit-learn", "React", "Node.js", "MongoDB", "Time Series Analysis"],

    links: { github: "#", demo: "#" },
    thumbnail: "/projects/daily-planner/thumbnail.jpg"
  },

  // ========================================
  // AI AGENTS & AUTOMATION
  // ========================================
  {
    id: "job-scraper",
    title: "Job Scraper & Outreach",
    domain: "agents",
    featured: true,

    timeline: "2 months",
    team: "Solo Project",
    status: "Active",
    year: "2024",

    tagline: "Automated Job Application Engine with AI Outreach",
    overview: "Intelligent automation system that scrapes job listings, filters by relevance, and drafts personalized outreach emails, achieving 10% interview rate across 500+ applications.",

    problem: "Finding and applying to relevant jobs is repetitive, time-consuming, and requires constant monitoring of multiple job boards.",

    solution: "Built end-to-end automation: web scraping with Selenium/BeautifulSoup, relevance scoring using NLP, and LLM-powered personalized email generation. System runs daily, tracks applications, and sends follow-ups.",

    results: {
      metrics: [
        { label: "Applications Sent", value: "500+", description: "Automated submissions" },
        { label: "Interview Rate", value: "10%", description: "50 interviews secured" },
        { label: "Time Saved", value: "200 hrs", description: "vs manual application" },
        { label: "Relevance Score", value: "85%", description: "Matched to profile" }
      ],
      impact: [
        "Secured 5 job offers through automated outreach",
        "Reduced application time from 30 min to 2 min per job",
        "Maintained personalized touch through AI customization"
      ]
    },

    tech: [
      "Python", "Selenium", "BeautifulSoup", "GPT-4", "LangChain",
      "SMTP", "PostgreSQL", "Cron Jobs"
    ],

    links: { github: "#", demo: "#" },
    thumbnail: "/projects/job-scraper/thumbnail.jpg"
  },

  {
    id: "discord-bot",
    title: "Discord Automation Tool",
    domain: "agents",
    featured: false,

    timeline: "1 month",
    team: "Solo Project",
    status: "Active",
    year: "2023",

    tagline: "24/7 Community Moderation Bot",
    overview: "Automated Discord bot for moderation, user onboarding, and role management, reducing manual workload by 60%.",

    problem: "Managing large Discord communities requires 24/7 moderation, which is unsustainable for volunteer moderators.",

    solution: "Built bot with auto-moderation (spam detection, profanity filter), automated welcome messages, role assignment based on activity, and custom commands.",

    results: {
      metrics: [
        { label: "Moderation Load Reduction", value: "60%", description: "Less manual work" },
        { label: "Community Size", value: "5K+", description: "Active members" },
        { label: "Uptime", value: "99.8%", description: "24/7 availability" }
      ],
      impact: [
        "Improved community engagement by 40%",
        "Faster response to rule violations",
        "Streamlined onboarding for new members"
      ]
    },

    tech: ["Python", "Discord.py", "SQLite", "Natural Language Processing"],

    links: { github: "#", demo: "#" },
    thumbnail: "/projects/discord-bot/thumbnail.jpg"
  },

  {
    id: "workflow-auto",
    title: "Python Workflow Automations",
    domain: "agents",
    featured: false,

    timeline: "Ongoing",
    team: "Solo Project",
    status: "Active",
    year: "2023-2024",

    tagline: "Custom Scripts for Daily Task Automation",
    overview: "Suite of Python automation scripts for file organization, data backups, and report generation, saving 5+ hours per week.",

    problem: "Repetitive daily technical tasks (file management, backups, report generation) reduce overall productivity.",

    solution: "Developed collection of automation scripts: smart file organizer, automated backup system, report generator, email parser, and data pipeline orchestrator.",

    results: {
      metrics: [
        { label: "Time Saved", value: "5+ hrs/week", description: "Automated tasks" },
        { label: "Error Reduction", value: "95%", description: "vs manual process" },
        { label: "Scripts Created", value: "15+", description: "Different automations" }
      ],
      impact: [
        "Eliminated human error in data handling",
        "Consistent execution of routine tasks",
        "More time for high-value work"
      ]
    },

    tech: ["Python", "Scripting", "Cron", "Pandas", "OS Automation"],

    links: { github: "#" },
    thumbnail: "/projects/workflow-auto/thumbnail.jpg"
  },

  // ========================================
  // AUTONOMOUS SYSTEMS
  // ========================================
  {
    id: "drone-swarm",
    title: "Drone Swarm Rescue",
    domain: "autonomous",
    featured: true,

    timeline: "4 months",
    team: "Solo Research",
    status: "Completed",
    year: "2023",

    tagline: "Coordinated Multi-Drone Search and Rescue System",
    overview: "Master-slave architecture enabling multiple drones to coordinate search zones efficiently, reducing search time by 40% in simulations with robust failure handling.",

    problem: "Search and rescue in large areas is slow with single agents. Manual coordination of multiple drones is complex and error-prone.",

    solution: "Developed distributed coordination system where master drone assigns search zones to slave drones based on area coverage optimization. Drones communicate findings in real-time and adapt to individual drone failures.",

    architecture: {
      description: "Distributed multi-agent system with centralized coordination",
      components: [
        {
          name: "Master Coordinator",
          description: "Assigns search zones, aggregates findings, handles drone failures"
        },
        {
          name: "Slave Drones",
          description: "Execute assigned search patterns, report findings, request reassignment"
        },
        {
          name: "Communication Layer",
          description: "ROS-based messaging for inter-drone coordination"
        },
        {
          name: "Path Planning",
          description: "A* algorithm for collision-free navigation"
        }
      ]
    },

    methodology: [
      "Simulation environment: Gazebo with custom search scenarios",
      "Drone count: 4 slave drones + 1 master",
      "Search area: 1km² with obstacles",
      "Communication: ROS topics with 10Hz update rate",
      "Failure scenarios: Battery depletion, communication loss, GPS errors"
    ],

    results: {
      metrics: [
        { label: "Search Time Reduction", value: "40%", description: "vs single drone" },
        { label: "Coverage Efficiency", value: "92%", description: "Area searched" },
        { label: "Failure Recovery", value: "100%", description: "Successful reassignments" },
        { label: "Coordination Overhead", value: "8%", description: "Communication cost" }
      ],
      impact: [
        "Demonstrated viability of swarm search",
        "Robust to individual drone failures",
        "Scalable to larger swarm sizes",
        "Potential for real-world SAR deployment"
      ]
    },

    challenges: [
      {
        challenge: "Communication Latency",
        solution: "Predictive position updates and local decision-making autonomy"
      },
      {
        challenge: "Zone Assignment Optimization",
        solution: "Greedy algorithm with dynamic reassignment based on progress"
      },
      {
        challenge: "Collision Avoidance",
        solution: "Distributed collision detection with priority-based resolution"
      }
    ],

    codeSnippets: [
      {
        title: "Master Coordinator Logic",
        language: "python",
        code: `class MasterCoordinator:
    def __init__(self, search_area, num_drones):
        self.search_area = search_area
        self.drones = self.initialize_drones(num_drones)
        self.zones = self.partition_area(search_area, num_drones)
    
    def assign_zones(self):
        """Assign search zones to available drones"""
        for drone, zone in zip(self.drones, self.zones):
            if drone.is_available():
                drone.assign_zone(zone)
                print(f"Assigned {zone} to {drone.id}")
    
    def handle_failure(self, failed_drone_id):
        """Reassign zone when drone fails"""
        failed_zone = self.get_zone(failed_drone_id)
        available_drone = self.find_nearest_available()
        
        if available_drone:
            available_drone.assign_zone(failed_zone)
            print(f"Reassigned {failed_zone} to {available_drone.id}")
    
    def aggregate_findings(self):
        """Collect and merge findings from all drones"""
        findings = []
        for drone in self.drones:
            findings.extend(drone.get_findings())
        return self.merge_overlapping(findings)`
      }
    ],

    tech: [
      "Python", "ROS (Robot Operating System)", "Gazebo", "DroneKit",
      "Path Planning Algorithms", "Multi-Agent Systems", "Simulation"
    ],

    links: {
      github: "#",
      demo: "#",
      documentation: "#"
    },

    thumbnail: "/projects/drone-swarm/thumbnail.jpg",
    images: [
      "/projects/drone-swarm/simulation.png",
      "/projects/drone-swarm/architecture.png"
    ]
  }
];

// Helper function to get projects by domain
export function getProjectsByDomain(domainId) {
  if (domainId === 'all') return projects;
  return projects.filter(p => p.domain === domainId);
}

// Helper function to get featured projects
export function getFeaturedProjects() {
  return projects.filter(p => p.featured);
}

// Helper function to get project by ID
export function getProjectById(id) {
  return projects.find(p => p.id === id);
}

"""
generate_defense_presentation.py
─────────────────────────────────
Generates a clean, classic, generic academic PowerPoint presentation for:
  Course: 21CSC305P Machine Learning
  Topic: Debate-Based Career Guidance Agent with Complementary Personas
  Focus: Units 3, 4, 5, and Capstone Multi-Persona Architecture
Format:
  - 16:9 Widescreen (13.333 in x 7.500 in)
  - Generic clean academic slide design: Solid Navy header bar, crisp white background
  - Clean bullet points, no emojis, professional typography (Calibri/Arial)
  - Real embedded plots from figures/ directory
  - Author credentials: Parth Gupta (RA2411003010019)
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ── Standard Academic Color Palette ───────────────────────────────────────────
COLOR_NAVY_DARK   = RGBColor(0, 33, 71)      # Deep Academic Navy #002147
COLOR_GOLD        = RGBColor(217, 119, 6)    # Accent Gold #D97706
COLOR_WHITE       = RGBColor(255, 255, 255)  # Pure White #FFFFFF
COLOR_TEXT_MAIN   = RGBColor(30, 41, 59)     # Slate-800 #1E293B
COLOR_TEXT_MUTED  = RGBColor(100, 116, 139)  # Slate-500 #64748B
COLOR_BORDER      = RGBColor(203, 213, 225)  # Slate-300 #CBD5E1
COLOR_BOX_BG      = RGBColor(248, 250, 252)  # Light Slate-50 #F8FAFC

SLIDE_WIDTH  = Inches(13.333)
SLIDE_HEIGHT = Inches(7.500)


def build_generic_presentation():
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT
    blank_layout = prs.slide_layouts[6]

    # Helper: Add Standard Generic Slide Template
    def add_content_slide(title_text, subtitle_text, slide_num):
        slide = prs.slides.add_slide(blank_layout)

        # White slide background
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_WHITE
        bg.line.fill.background()

        # Top Header Bar (Solid Navy)
        header_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, Inches(1.15))
        header_bar.fill.solid()
        header_bar.fill.fore_color.rgb = COLOR_NAVY_DARK
        header_bar.line.fill.background()

        # Gold Underline Accent
        gold_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.15), SLIDE_WIDTH, Inches(0.04))
        gold_line.fill.solid()
        gold_line.fill.fore_color.rgb = COLOR_GOLD
        gold_line.line.fill.background()

        # Header Title Text Box
        tbox = slide.shapes.add_textbox(Inches(0.8), Inches(0.15), Inches(11.733), Inches(0.95))
        tf = tbox.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_title = tf.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = "Calibri"
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_WHITE

        if subtitle_text:
            p_sub = tf.add_paragraph()
            p_sub.text = subtitle_text
            p_sub.font.name = "Calibri"
            p_sub.font.size = Pt(12)
            p_sub.font.color.rgb = RGBColor(226, 232, 240)
            p_sub.space_before = Pt(3)

        # Bottom Footer Line
        fline = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(6.95), Inches(11.733), Inches(0.015))
        fline.fill.solid()
        fline.fill.fore_color.rgb = COLOR_BORDER
        fline.line.fill.background()

        # Footer Text
        fbox = slide.shapes.add_textbox(Inches(0.8), Inches(7.02), Inches(11.733), Inches(0.35))
        ftf = fbox.text_frame
        ftf.word_wrap = True
        ftf.margin_left = ftf.margin_top = ftf.margin_right = ftf.margin_bottom = 0
        p_foot = ftf.paragraphs[0]
        p_foot.text = f"SRM Institute of Science and Technology  |  Department of Computing Technologies  |  21CSC305P Machine Learning  |  Slide {slide_num}"
        p_foot.font.name = "Calibri"
        p_foot.font.size = Pt(9.5)
        p_foot.font.color.rgb = COLOR_TEXT_MUTED

        return slide

    # ── SLIDE 1: Standard Academic Title Slide ────────────────────────────────
    slide1 = prs.slides.add_slide(blank_layout)

    # Clean White Background
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_WHITE
    bg1.line.fill.background()

    # Top Navy Banner Strip
    top_strip = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, Inches(0.2))
    top_strip.fill.solid()
    top_strip.fill.fore_color.rgb = COLOR_NAVY_DARK
    top_strip.line.fill.background()

    # Institution Header
    inst_box = slide1.shapes.add_textbox(Inches(1.0), Inches(0.7), Inches(11.333), Inches(0.9))
    itf = inst_box.text_frame
    itf.word_wrap = True
    itf.margin_left = itf.margin_top = itf.margin_right = itf.margin_bottom = 0
    p_inst = itf.paragraphs[0]
    p_inst.text = "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY"
    p_inst.font.name = "Calibri"
    p_inst.font.size = Pt(16)
    p_inst.font.bold = True
    p_inst.font.color.rgb = COLOR_NAVY_DARK

    p_dept = itf.add_paragraph()
    p_dept.text = "Department of Computing Technologies  •  College of Engineering and Technology"
    p_dept.font.name = "Calibri"
    p_dept.font.size = Pt(13)
    p_dept.font.color.rgb = COLOR_TEXT_MUTED
    p_dept.space_before = Pt(3)

    # Accent Gold Line
    t_line = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.7), Inches(11.333), Inches(0.03))
    t_line.fill.solid()
    t_line.fill.fore_color.rgb = COLOR_GOLD
    t_line.line.fill.background()

    # Project Title Box
    title_box = slide1.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(11.333), Inches(2.2))
    ttf = title_box.text_frame
    ttf.word_wrap = True
    ttf.margin_left = ttf.margin_top = ttf.margin_right = ttf.margin_bottom = 0

    p_main = ttf.paragraphs[0]
    p_main.text = "DEBATE-BASED CAREER GUIDANCE AGENT WITH COMPLEMENTARY PERSONAS"
    p_main.font.name = "Calibri"
    p_main.font.size = Pt(28)
    p_main.font.bold = True
    p_main.font.color.rgb = COLOR_NAVY_DARK

    p_sub = ttf.add_paragraph()
    p_sub.text = "Comprehensive Algorithmic Synthesis & Capstone Defense (Units 3, 4, and 5)"
    p_sub.font.name = "Calibri"
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = COLOR_GOLD
    p_sub.space_before = Pt(8)

    p_course = ttf.add_paragraph()
    p_course.text = "Course: 21CSC305P - Machine Learning  |  Academic Year: 2026–2027"
    p_course.font.name = "Calibri"
    p_course.font.size = Pt(13)
    p_course.font.color.rgb = COLOR_TEXT_MAIN
    p_course.space_before = Pt(6)

    # Divider Line
    m_line = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(4.5), Inches(11.333), Inches(0.015))
    m_line.fill.solid()
    m_line.fill.fore_color.rgb = COLOR_BORDER
    m_line.line.fill.background()

    # Authors and Guides in 2 generic columns
    author_box = slide1.shapes.add_textbox(Inches(1.0), Inches(4.7), Inches(5.5), Inches(2.2))
    atf = author_box.text_frame
    atf.word_wrap = True
    atf.margin_left = atf.margin_top = atf.margin_right = atf.margin_bottom = 0
    p = atf.paragraphs[0]
    p.text = "Submitted by:"
    p.font.name = "Calibri"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    students = [
        "Parth Gupta (RA2411003010019)",
        "Mokshith Penna (RA2411003010044)",
        "Ishu Gurtu (RA2411003010015)"
    ]
    for s in students:
        p_s = atf.add_paragraph()
        p_s.text = f"• {s}"
        p_s.font.name = "Calibri"
        p_s.font.size = Pt(13)
        p_s.font.color.rgb = COLOR_TEXT_MAIN
        p_s.space_before = Pt(4)

    guide_box = slide1.shapes.add_textbox(Inches(6.8), Inches(4.7), Inches(5.5), Inches(2.2))
    gtf = guide_box.text_frame
    gtf.word_wrap = True
    gtf.margin_left = gtf.margin_top = gtf.margin_right = gtf.margin_bottom = 0
    p = gtf.paragraphs[0]
    p.text = "Under the Guidance of:"
    p.font.name = "Calibri"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    guides = [
        "Dr. Divya Mohan, Associate Professor (Project Guide)",
        "Dr. G. Niranjana, Professor & Head of Department"
    ]
    for g in guides:
        p_g = gtf.add_paragraph()
        p_g.text = f"• {g}"
        p_g.font.name = "Calibri"
        p_g.font.size = Pt(13)
        p_g.font.color.rgb = COLOR_TEXT_MAIN
        p_g.space_before = Pt(4)

    # Bottom Navy Banner Strip
    bot_strip = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.3), SLIDE_WIDTH, Inches(0.2))
    bot_strip.fill.solid()
    bot_strip.fill.fore_color.rgb = COLOR_NAVY_DARK
    bot_strip.line.fill.background()

    # ── SLIDE 2: Project Overview & System Architecture ──────────────────────
    slide2 = add_content_slide(
        "Project Overview & System Architecture",
        "Integration of Curricular Units 3, 4, and 5 into an Adversarial Debate System",
        2
    )
    # Left text column
    tbox2 = slide2.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tf2 = tbox2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = 0

    p = tf2.paragraphs[0]
    p.text = "Problem Statement & Motivation:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    p_desc = tf2.add_paragraph()
    p_desc.text = "Conventional career advisement systems rely on static single-model classifications that provide no uncertainty estimates, cannot explain decisions, and frequently fail on borderline students."
    p_desc.font.name = "Calibri"
    p_desc.font.size = Pt(12)
    p_desc.font.color.rgb = COLOR_TEXT_MAIN
    p_desc.space_before = Pt(4)

    p_arch = tf2.add_paragraph()
    p_arch.text = "Curricular Unit Integration:"
    p_arch.font.name = "Calibri"
    p_arch.font.size = Pt(14)
    p_arch.font.bold = True
    p_arch.font.color.rgb = COLOR_NAVY_DARK
    p_arch.space_before = Pt(10)

    units_data = [
        ("Unit 3 (Unsupervised Learning)", "PCA dimensionality reduction (6 factors) + GMM Expectation-Maximization soft clustering + Mahalanobis distance outlier detection."),
        ("Unit 4 (Sequential Dynamics)", "5-state longitudinal Markov chain (A^3) + Scaled HMM with Viterbi MAP path decoding + Kalman filter growth tracking."),
        ("Unit 5 (Ensemble Learning)", "CART decision trees with pruning + Random Forest bagging + AdaBoost sequential reweighting + GBDT gradient descent."),
        ("Capstone Architecture", "Feeds outputs into 3 adversarial personas (Planner, Critic, Sentinel) adjudicated by a mathematical Arbiter.")
    ]
    for u_t, u_d in units_data:
        p_u = tf2.add_paragraph()
        p_u.text = f"• {u_t}: {u_d}"
        p_u.font.name = "Calibri"
        p_u.font.size = Pt(11.5)
        p_u.font.color.rgb = COLOR_TEXT_MAIN
        p_u.space_before = Pt(6)

    # Right Image
    if os.path.exists("figures/capstone_debate_architecture.png"):
        pic = slide2.shapes.add_picture("figures/capstone_debate_architecture.png", Inches(6.7), Inches(1.5), width=Inches(5.8))
        pic.line.color.rgb = COLOR_BORDER
        pic.line.width = Pt(1)

    # ── SLIDE 3: Unit 3 — PCA & Dimensionality Reduction ─────────────────────
    slide3 = add_content_slide(
        "Unit 3: Principal Component Analysis (PCA)",
        "Dimensionality Reduction and Latent Competency Factor Extraction",
        3
    )
    tbox3 = slide3.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tf3 = tbox3.text_frame
    tf3.word_wrap = True
    tf3.margin_left = tf3.margin_top = tf3.margin_right = tf3.margin_bottom = 0

    p = tf3.paragraphs[0]
    p.text = "Mathematical Formulation:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets3 = [
        ("Eigen-System Decomposition", "Computes covariance matrix across 14 standardized competency variables and solves Sigma * v_i = lambda_i * v_i."),
        ("Kaiser Criterion (lambda > 1.0)", "Retains exclusively components with eigenvalues greater than 1.0, eliminating uninformative noise dimensions."),
        ("Identified Latent Factors (6 Components)", "Six principal components capture 43.89% cumulative variance across 20,000 student records:\n- PC1 (20.3%): Core Computer Science & Algorithms\n- PC2 (8.4%): Mathematics & Analytical Reasoning\n- PC3 (5.1%): Hardware & Computer Architecture\n- PC4-PC6 (10.1%): Networking, Soft Skills, Leadership"),
        ("Role in Agent Architecture", "The Strategic Career Planner projects raw student vectors into PC1-PC2 coordinates to visualize alignment with career benchmarks.")
    ]
    for b_t, b_d in bullets3:
        p_b = tf3.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    if os.path.exists("figures/unit3_pca_scree.png"):
        pic = slide3.shapes.add_picture("figures/unit3_pca_scree.png", Inches(6.7), Inches(1.5), width=Inches(5.8))
        pic.line.color.rgb = COLOR_BORDER
        pic.line.width = Pt(1)

    # ── SLIDE 4: Unit 3 — Gaussian Mixture Models & EM Algorithm ─────────────
    slide4 = add_content_slide(
        "Unit 3: Gaussian Mixture Models & EM Algorithm",
        "Soft Probabilistic Clustering and Assignment Uncertainty Quantification",
        4
    )
    tbox4 = slide4.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tf4 = tbox4.text_frame
    tf4.word_wrap = True
    tf4.margin_left = tf4.margin_top = tf4.margin_right = tf4.margin_bottom = 0

    p = tf4.paragraphs[0]
    p.text = "Expectation-Maximization Formulation (Bishop §9.2):"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets4 = [
        ("Mixture Model Setup", "Assumes K=4 latent career archetypes governed by hidden categorical indicator Z ~ Categorical(pi) with full covariance matrices Sigma_k."),
        ("Expectation Step (E-Step)", "Computes soft posterior responsibilities:\ngamma_nk = P(Z_k=1 | x_n) = [pi_k * N(x_n | mu_k, Sigma_k)] / [sum pi_j * N(x_n | mu_j, Sigma_j)]."),
        ("Maximization Step (M-Step)", "Iteratively updates cluster centers mu_k, covariance matrices Sigma_k, and mixing coefficients pi_k to maximize log-likelihood."),
        ("Shannon Assignment Entropy", "H(gamma) = - sum gamma_nk * ln(gamma_nk). Directly measures classification ambiguity across career tracks."),
        ("Ethics Sentinel Safety Guardrail", "When H(gamma) exceeds threshold tau = 1.20 nats, candidate profile is deemed ambiguous, triggering Human Counselor Override.")
    ]
    for b_t, b_d in bullets4:
        p_b = tf4.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(7)

    if os.path.exists("figures/unit3_gmm_responsibilities_heatmap.png"):
        pic = slide4.shapes.add_picture("figures/unit3_gmm_responsibilities_heatmap.png", Inches(6.7), Inches(1.5), width=Inches(5.8))
        pic.line.color.rgb = COLOR_BORDER
        pic.line.width = Pt(1)

    # ── SLIDE 5: Unit 3 — Mahalanobis Distance & Outlier Detection ───────────
    slide5 = add_content_slide(
        "Unit 3: Mahalanobis Distance & Cluster Benchmark",
        "Covariance-Whitened Anomaly Detection and Separation Metrics",
        5
    )
    tbox5 = slide5.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tf5 = tbox5.text_frame
    tf5.word_wrap = True
    tf5.margin_left = tf5.margin_top = tf5.margin_right = tf5.margin_bottom = 0

    p = tf5.paragraphs[0]
    p.text = "Mathematical Formulation & Benchmarking:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets5 = [
        ("Mahalanobis Distance Metric", "D_M(x, mu) = sqrt((x - mu)^T * Sigma^-1 * (x - mu)). Accounts for correlations and variances across competency dimensions."),
        ("Why Euclidean Distance Fails", "Euclidean distance assumes independent, spherical variance, double-counting highly correlated skills (e.g., Operating Systems and Algorithms)."),
        ("Empirical Critic Outlier Threshold", "Cutoff threshold set at tau = 3.50. Profiles with D_M > 3.50 are flagged as anomaly outliers, preventing inaccurate career recommendations."),
        ("Unsupervised Benchmark Comparison", "Evaluated across 20,000 instances:\n- K-Means: Silhouette = 0.0482, Calinski-Harabasz = 874.2\n- GMM (EM): Silhouette = 0.0431, Calinski-Harabasz = 812.5 (enables soft entropy)\n- Hierarchical (Ward): Silhouette = 0.0115, Calinski-Harabasz = 645.1")
    ]
    for b_t, b_d in bullets5:
        p_b = tf5.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    if os.path.exists("figures/unit3_clustering_comparison.png"):
        pic = slide5.shapes.add_picture("figures/unit3_clustering_comparison.png", Inches(6.7), Inches(1.5), width=Inches(5.8))
        pic.line.color.rgb = COLOR_BORDER
        pic.line.width = Pt(1)

    # ── SLIDE 6: Unit 4 — Markov Chains & Transition Dynamics ────────────────
    slide6 = add_content_slide(
        "Unit 4: Sequential Career Modeling with Markov Chains",
        "Multi-Step Chapman-Kolmogorov Transitions and Bottleneck Analysis",
        6
    )
    tbox6 = slide6.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tf6 = tbox6.text_frame
    tf6.word_wrap = True
    tf6.margin_left = tf6.margin_top = tf6.margin_right = tf6.margin_bottom = 0

    p = tf6.paragraphs[0]
    p.text = "Discrete Markovian Dynamics:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets6 = [
        ("5-State Career Hierarchy", "S1: Junior Associate -> S2: Mid-Level Engineer -> S3: Senior Specialist -> S4: Lead Architect -> S5: Director / CTO."),
        ("Multi-Step Transition Matrix (A^3)", "Computes 3-year cumulative transition probabilities P(X_{t+3} = j | X_t = i) via matrix power A^3 = A x A x A."),
        ("Stationary Distribution (pi*)", "Solves the balance equation pi* A = pi* subject to sum pi_i = 1, verifying long-term absorbing characteristics."),
        ("Empirical Critic Bottleneck Audit", "Identifies a 40% retention plateau at the Mid-to-Senior transition (P_stay = 0.40). Students lacking project experience risk early career stagnation."),
        ("Remedial Action", "The system recommends targeted technical portfolio development during Semesters 3 and 4 to overcome this plateau.")
    ]
    for b_t, b_d in bullets6:
        p_b = tf6.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    if os.path.exists("figures/unit4_markov_transition_dynamics.png"):
        pic = slide6.shapes.add_picture("figures/unit4_markov_transition_dynamics.png", Inches(6.7), Inches(1.5), width=Inches(5.8))
        pic.line.color.rgb = COLOR_BORDER
        pic.line.width = Pt(1)

    # ── SLIDE 7: Unit 4 — Hidden Markov Models & Viterbi Decoding ─────────────
    slide7 = add_content_slide(
        "Unit 4: Hidden Markov Models & Viterbi Decoding",
        "Maximum A Posteriori (MAP) Career Progression Sequence Estimation",
        7
    )
    tbox7 = slide7.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tf7 = tbox7.text_frame
    tf7.word_wrap = True
    tf7.margin_left = tf7.margin_top = tf7.margin_right = tf7.margin_bottom = 0

    p = tf7.paragraphs[0]
    p.text = "HMM Formulation & Trellis Decoding:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets7 = [
        ("Latent State Setup", "Hidden competency maturity states Z_t in {Novice, Competent, Expert} emit observable quarterly evaluation scores X_t."),
        ("Scaled Forward-Backward Algorithm", "Computes sequence log-likelihood ln P(X | lambda) = -4.410 using dynamic scaling to eliminate numerical underflow."),
        ("Viterbi Dynamic Programming Recursion", "Computes MAP path:\nv_t(j) = max_i [v_{t-1}(i) * a_ij] * b_j(x_t), with back-pointers psi_t(j) = argmax_i [v_{t-1}(i) * a_ij]."),
        ("Optimal Progression Trajectory", "Decoded optimal milestone sequence:\nSem 1 (Novice) -> Sem 2 (Competent) -> Sem 3 (Competent) -> Sem 4 (Expert) -> Sem 5 (Expert)."),
        ("Role in Guidance Agent", "Supplies the Strategic Career Planner with an objective, step-by-step roadmap to guide student skill acquisition.")
    ]
    for b_t, b_d in bullets7:
        p_b = tf7.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    if os.path.exists("figures/unit4_workflow.png"):
        pic = slide7.shapes.add_picture("figures/unit4_workflow.png", Inches(6.7), Inches(1.5), width=Inches(5.8))
        pic.line.color.rgb = COLOR_BORDER
        pic.line.width = Pt(1)

    # ── SLIDE 8: Unit 4 — Kalman Filtering for Skill Tracking ─────────────────
    slide8 = add_content_slide(
        "Unit 4: Linear Dynamical Systems & Kalman Filter",
        "Recursive State-Space Estimation of True Latent Competency Velocity",
        8
    )
    tbox8 = slide8.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tf8 = tbox8.text_frame
    tf8.word_wrap = True
    tf8.margin_left = tf8.margin_top = tf8.margin_right = tf8.margin_bottom = 0

    p = tf8.paragraphs[0]
    p.text = "Kalman Filter Mathematical Mechanics (Murphy §18):"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets8 = [
        ("State Formulation", "State vector x_t = [Proficiency_t, Velocity_t]^T with transition matrix F = [[1, Delta_t], [0, 1]]."),
        ("Measurement Model", "Observations z_t = H x_t + v_t, where H = [1, 0] and observation noise v_t ~ N(0, R)."),
        ("State Prediction & Covariance", "x_pred = F * x_hat_{t-1}, P_pred = F * P_{t-1} * F^T + Q."),
        ("Kalman Gain & Measurement Update", "K_t = P_pred * H^T * (H * P_pred * H^T + R)^-1.\nx_hat_t = x_pred + K_t * (z_t - H * x_pred), P_t = (I - K_t * H) * P_pred."),
        ("Academic Rationale", "Filters temporary semester score fluctuations, providing the Planner persona with estimated growth velocity (+1.2 proficiency units/quarter).")
    ]
    for b_t, b_d in bullets8:
        p_b = tf8.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    if os.path.exists("figures/unit4_kalman_filter.png"):
        pic = slide8.shapes.add_picture("figures/unit4_kalman_filter.png", Inches(6.7), Inches(1.5), width=Inches(5.8))
        pic.line.color.rgb = COLOR_BORDER
        pic.line.width = Pt(1)

    # ── SLIDE 9: Unit 5 — Decision Trees (CART) & Pruning ────────────────────
    slide9 = add_content_slide(
        "Unit 5: Decision Trees & Minimal Cost-Complexity Pruning",
        "Controlling High Variance via Regularized Tree Pruning",
        9
    )
    tbox9 = slide9.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tf9 = tbox9.text_frame
    tf9.word_wrap = True
    tf9.margin_left = tf9.margin_top = tf9.margin_right = tf9.margin_bottom = 0

    p = tf9.paragraphs[0]
    p.text = "Tree Splitting & Pruning Mechanics:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets9 = [
        ("Gini Impurity Splitting", "I_G(t) = 1 - sum p_tk^2. Greedily selects feature j and split threshold s maximizing impurity drop Delta I_G."),
        ("Variance in Deep Trees", "Unconstrained decision trees achieve near-zero training error but exhibit severe overfitting on test datasets."),
        ("Minimal Cost-Complexity Pruning", "R_alpha(T) = R(T) + alpha * |T|. Penalizes the total number of terminal leaves |T| using regularizer alpha >= 0."),
        ("Pruning Path Optimization", "Evaluates effective alphas along the pruning path, identifying optimal maximum depth = 5 to maximize generalization accuracy."),
        ("Interpretability Value", "Generates human-auditable decision rules, allowing academic counselors to verify recommendations directly.")
    ]
    for b_t, b_d in bullets9:
        p_b = tf9.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    if os.path.exists("figures/unit5_workflow.png"):
        pic = slide9.shapes.add_picture("figures/unit5_workflow.png", Inches(6.7), Inches(1.5), width=Inches(5.8))
        pic.line.color.rgb = COLOR_BORDER
        pic.line.width = Pt(1)

    # ── SLIDE 10: Unit 5 — Random Forest vs AdaBoost ─────────────────────────
    slide10 = add_content_slide(
        "Unit 5: Bagging vs Boosting Paradigms",
        "Random Forest Variance Reduction vs AdaBoost Sequential Loss Minimization",
        10
    )
    # 2 Comparison Columns
    col_w = Inches(5.6)

    # Box 1: Random Forest
    b1 = slide10.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), col_w, Inches(5.2))
    b1.fill.solid()
    b1.fill.fore_color.rgb = COLOR_BOX_BG
    b1.line.color.rgb = COLOR_NAVY_DARK
    b1.line.width = Pt(1.5)
    btf1 = b1.text_frame
    btf1.word_wrap = True
    btf1.margin_left = btf1.margin_right = Inches(0.25)
    btf1.margin_top = Inches(0.2)

    p1 = btf1.paragraphs[0]
    p1.text = "Random Forest (Bagging Paradigm)"
    p1.font.name = "Calibri"
    p1.font.size = Pt(15)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY_DARK

    rf_bullets = [
        "Bootstrap Aggregation: Builds B=100 trees on bootstrap samples with replacement (~63.2% unique instances).",
        "Feature Subspacing: Evaluates m = sqrt(D) ~ 3 random features at each split to decorrelate individual trees.",
        "Variance Reduction: Var(T_bar) = rho * sigma^2 + [(1 - rho)/B] * sigma^2. Tree decorrelation drives total variance down.",
        "Out-Of-Bag (OOB) Validation: Evaluates generalization on unused instances (OOB Score = 28.5%).",
        "Performance: Achieved 34.90% weighted precision, making it the most reliable predictor across all 5 macro tracks."
    ]
    for pt in rf_bullets:
        p_pt = btf1.add_paragraph()
        p_pt.text = f"• {pt}"
        p_pt.font.name = "Calibri"
        p_pt.font.size = Pt(11.5)
        p_pt.font.color.rgb = COLOR_TEXT_MAIN
        p_pt.space_before = Pt(8)

    # Box 2: AdaBoost
    b2 = slide10.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.5), col_w, Inches(5.2))
    b2.fill.solid()
    b2.fill.fore_color.rgb = COLOR_BOX_BG
    b2.line.color.rgb = COLOR_GOLD
    b2.line.width = Pt(1.5)
    btf2 = b2.text_frame
    btf2.word_wrap = True
    btf2.margin_left = btf2.margin_right = Inches(0.25)
    btf2.margin_top = Inches(0.2)

    p2 = btf2.paragraphs[0]
    p2.text = "AdaBoost (Boosting Paradigm)"
    p2.font.name = "Calibri"
    p2.font.size = Pt(15)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_GOLD

    ada_bullets = [
        "Sequential Stumps: Iteratively trains weak decision stumps focusing on previously misclassified student profiles.",
        "Exponential Loss: Minimizes L(y, f) = exp(-y * f(x)) across training iterations.",
        "Sample Reweighting: Increases weights of misclassified instances: w_i^(m+1) = w_i^(m) * exp(alpha_m * I(y_i != h_m(x_i))).",
        "Weak Learner Weight: alpha_m = 0.5 * ln[(1 - eps_m) / eps_m].",
        "Sensitivity to Outliers: High sensitivity to noisy boundary profiles; sample weights explode on ambiguous student records."
    ]
    for pt in ada_bullets:
        p_pt = btf2.add_paragraph()
        p_pt.text = f"• {pt}"
        p_pt.font.name = "Calibri"
        p_pt.font.size = Pt(11.5)
        p_pt.font.color.rgb = COLOR_TEXT_MAIN
        p_pt.space_before = Pt(8)

    # ── SLIDE 11: Unit 5 — GBDT & Algorithmic Benchmark ──────────────────────
    slide11 = add_content_slide(
        "Unit 5: Gradient Boosting & Algorithmic Benchmark",
        "Functional Gradient Descent and Comprehensive Multi-Model Evaluation",
        11
    )
    tbox11 = slide11.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tf11 = tbox11.text_frame
    tf11.word_wrap = True
    tf11.margin_left = tf11.margin_top = tf11.margin_right = tf11.margin_bottom = 0

    p = tf11.paragraphs[0]
    p.text = "Gradient Boosted Decision Trees (GBDT):"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets11 = [
        ("Pseudo-Residual Optimization", "Fits regression trees directly to the negative gradients of multi-class log loss: r_ikm = -[dL(y_i, f_k(x_i)) / df_k(x_i)]."),
        ("Shrinkage Regularization (eta = 0.1)", "Dampens step sizes in function space to prevent overfitting training anomalies."),
        ("Multi-Model Comparative Benchmark", "Tested on 4,000 held-out student records across 5 macro tracks:\n- Single Decision Tree: Accuracy 27.84%, F1 0.1540\n- AdaBoost: Accuracy 28.24%, F1 0.1308\n- Random Forest: Accuracy 28.66%, Precision 34.90% (Top Performer)\n- Gradient Boosted Trees: Accuracy 26.50%, F1 0.1721"),
        ("Key Influential Competencies", "Algorithms (0.135), Operating Systems (0.128), and Programming Concepts (0.121) have highest Gini feature importance.")
    ]
    for b_t, b_d in bullets11:
        p_b = tf11.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    if os.path.exists("figures/unit5_ensemble_comparison.png"):
        pic = slide11.shapes.add_picture("figures/unit5_ensemble_comparison.png", Inches(6.7), Inches(1.5), width=Inches(5.8))
        pic.line.color.rgb = COLOR_BORDER
        pic.line.width = Pt(1)

    # ── SLIDE 12: Capstone — Multi-Persona Debate Architecture ────────────────
    slide12 = add_content_slide(
        "Integrated Capstone: Multi-Persona Guidance Architecture",
        "Adversarial Personas with Mathematical Safety Guardrails",
        12
    )
    # 3 Persona Cards
    p_w = Inches(3.7)
    x_coords = [Inches(0.8), Inches(4.8), Inches(8.8)]

    # Persona 1: Planner
    c1 = slide12.shapes.add_shape(MSO_SHAPE.RECTANGLE, x_coords[0], Inches(1.5), p_w, Inches(3.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = COLOR_BOX_BG
    c1.line.color.rgb = COLOR_NAVY_DARK
    c1.line.width = Pt(1.5)
    ctf1 = c1.text_frame
    ctf1.word_wrap = True
    ctf1.margin_left = ctf1.margin_right = Inches(0.2)
    ctf1.margin_top = Inches(0.15)
    p = ctf1.paragraphs[0]
    p.text = "Strategic Career Planner\n(Growth & Trajectory)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK
    pts1 = [
        "Unit 2: Analytical Bayesian Regression predicted compensation (mu_N) with 95% credible intervals.",
        "Unit 3: GMM component membership probability fit.",
        "Unit 4: Viterbi MAP career advancement trajectory.",
        "Role: Charts ambitious upward career path for qualified students."
    ]
    for pt in pts1:
        p_pt = ctf1.add_paragraph()
        p_pt.text = f"• {pt}"
        p_pt.font.name = "Calibri"
        p_pt.font.size = Pt(10.5)
        p_pt.font.color.rgb = COLOR_TEXT_MAIN
        p_pt.space_before = Pt(6)

    # Persona 2: Critic
    c2 = slide12.shapes.add_shape(MSO_SHAPE.RECTANGLE, x_coords[1], Inches(1.5), p_w, Inches(3.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = COLOR_BOX_BG
    c2.line.color.rgb = COLOR_GOLD
    c2.line.width = Pt(1.5)
    ctf2 = c2.text_frame
    ctf2.word_wrap = True
    ctf2.margin_left = ctf2.margin_right = Inches(0.2)
    ctf2.margin_top = Inches(0.15)
    p = ctf2.paragraphs[0]
    p.text = "Empirical Critic\n(Feasibility & Metrics)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_GOLD
    pts2 = [
        "Unit 3: Mahalanobis distance D_M vs outlier threshold tau = 3.50.",
        "Unit 4: Markov multi-step transition A^3 retention bottleneck audit.",
        "Prerequisite verification: flags specific deficient competency dimensions.",
        "Role: Prevents student failure by challenging unrealistic goals."
    ]
    for pt in pts2:
        p_pt = ctf2.add_paragraph()
        p_pt.text = f"• {pt}"
        p_pt.font.name = "Calibri"
        p_pt.font.size = Pt(10.5)
        p_pt.font.color.rgb = COLOR_TEXT_MAIN
        p_pt.space_before = Pt(6)

    # Persona 3: Sentinel
    c3 = slide12.shapes.add_shape(MSO_SHAPE.RECTANGLE, x_coords[2], Inches(1.5), p_w, Inches(3.8))
    c3.fill.solid()
    c3.fill.fore_color.rgb = COLOR_BOX_BG
    c3.line.color.rgb = RGBColor(220, 38, 38)
    c3.line.width = Pt(1.5)
    ctf3 = c3.text_frame
    ctf3.word_wrap = True
    ctf3.margin_left = ctf3.margin_right = Inches(0.2)
    ctf3.margin_top = Inches(0.15)
    p = ctf3.paragraphs[0]
    p.text = "Ethics & Fairness Sentinel\n(Uncertainty & Safety)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(220, 38, 38)
    pts3 = [
        "Unit 3: GMM Assignment Shannon Entropy H(gamma) vs cutoff tau = 1.20 nats.",
        "Automated Human Counselor Override engaged upon high multi-modal ambiguity.",
        "Demographic parity audit across all career tracks.",
        "Role: Protects students from biased or ambiguous algorithmic recommendations."
    ]
    for pt in pts3:
        p_pt = ctf3.add_paragraph()
        p_pt.text = f"• {pt}"
        p_pt.font.name = "Calibri"
        p_pt.font.size = Pt(10.5)
        p_pt.font.color.rgb = COLOR_TEXT_MAIN
        p_pt.space_before = Pt(6)

    # Arbiter Bottom Box
    abox = slide12.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(5.5), Inches(11.7), Inches(1.25))
    abox.fill.solid()
    abox.fill.fore_color.rgb = COLOR_BOX_BG
    abox.line.color.rgb = RGBColor(22, 163, 74)
    abox.line.width = Pt(1.5)
    atf = abox.text_frame
    atf.word_wrap = True
    atf.margin_left = atf.margin_right = Inches(0.25)
    atf.margin_top = Inches(0.12)
    p = atf.paragraphs[0]
    p.text = "Arbiter Consensus Synthesis"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(22, 163, 74)

    p_a = atf.add_paragraph()
    p_a.text = "Balances the Strategic Planner's growth trajectory against the Empirical Critic's feasibility checks and the Ethics Sentinel's uncertainty guardrails. Synthesizes a calibrated confidence score (65% to 96%), target career milestone, and a personalized 4-semester remedial coursework plan."
    p_a.font.name = "Calibri"
    p_a.font.size = Pt(11)
    p_a.font.color.rgb = COLOR_TEXT_MAIN
    p_a.space_before = Pt(3)

    # ── SLIDE 13: Capstone — Case Studies & Guardrail Verification ────────────
    slide13 = add_content_slide(
        "Capstone: Case Studies & Guardrail Verification",
        "Empirical Validation Across 4 Representative Student Profiles",
        13
    )
    # 4 generic cards
    card_w13 = Inches(5.7)
    card_h13 = Inches(2.4)
    coords = [
        (Inches(0.8), Inches(1.5)),
        (Inches(6.8), Inches(1.5)),
        (Inches(0.8), Inches(4.25)),
        (Inches(6.8), Inches(4.25))
    ]
    scenarios = [
        ("Case Study 1: High Achiever (Parth Gupta - RA2411003010019)",
         "Academic Profile: Algorithms 96%, Operating Systems 95%, Coding 10/10.\n- Mahalanobis Distance: D_M = 1.82 (Well within alumni core)\n- Shannon Entropy: H = 0.42 nats (Very sharp classification)\n- Result: Unanimous Dialectical Consensus (96% Confidence)\n- Action: Accelerated track into AI / Machine Learning Specialist.",
         COLOR_NAVY_DARK),

        ("Case Study 2: Skill Deficit Anomaly",
         "Academic Profile: Mathematics 51%, Algorithms 52%, Logic 4/10.\n- Mahalanobis Distance: D_M = 4.50 > 3.50 (Outlier Anomaly)\n- Critic Stance: Risk Flagged — Critical Prerequisite Deficit\n- Result: Conditional Advancement (81% Confidence)\n- Action: Mandatory 2-semester prerequisite skill remediation.",
         COLOR_GOLD),

        ("Case Study 3: Ambiguous Boundary Profile",
         "Academic Profile: Uniform 74% across all subjects.\n- Shannon Entropy: H = 1.35 > 1.20 nats (High Epistemic Ambiguity)\n- Ethics Sentinel: Human Counselor Override Engaged\n- Result: Hybrid Counselor Mediation (65% Confidence)\n- Action: Referral to Department Academic Counseling Cell.",
         RGBColor(220, 38, 38)),

        ("Case Study 4: Cloud & Systems Architect",
         "Academic Profile: Computer Networks 96%, Operating Systems 92%.\n- Mahalanobis Distance: D_M = 2.30, Shannon Entropy: H = 0.72 nats\n- All Personas Validate Competency Fit\n- Result: Full Consensus (91% Confidence)\n- Action: Enterprise cloud architecture certification & placement.",
         RGBColor(22, 163, 74))
    ]

    for i, (s_title, s_text, border_c) in enumerate(scenarios):
        sc_shape = slide13.shapes.add_shape(MSO_SHAPE.RECTANGLE, coords[i][0], coords[i][1], card_w13, card_h13)
        sc_shape.fill.solid()
        sc_shape.fill.fore_color.rgb = COLOR_BOX_BG
        sc_shape.line.color.rgb = border_c
        sc_shape.line.width = Pt(1.5)

        stf = sc_shape.text_frame
        stf.word_wrap = True
        stf.margin_left = stf.margin_right = Inches(0.25)
        stf.margin_top = Inches(0.15)
        p = stf.paragraphs[0]
        p.text = s_title
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = border_c

        p_desc = stf.add_paragraph()
        p_desc.text = s_text
        p_desc.font.name = "Calibri"
        p_desc.font.size = Pt(11)
        p_desc.font.color.rgb = COLOR_TEXT_MAIN
        p_desc.space_before = Pt(4)

    # ── SLIDE 14: Conclusion & Academic References ────────────────────────────
    slide14 = add_content_slide(
        "Conclusion & Academic References",
        "Key Project Takeaways and Verified Theoretical Literature",
        14
    )
    # Left column: Conclusions
    tbox14 = slide14.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tf14 = tbox14.text_frame
    tf14.word_wrap = True
    tf14.margin_left = tf14.margin_top = tf14.margin_right = tf14.margin_bottom = 0

    p = tf14.paragraphs[0]
    p.text = "Key Project Conclusions:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    conclusions = [
        ("Multi-Persona Guidance", "Adversarial dialectical debate successfully eliminates single-model overconfidence by balancing ambition, feasibility, and safety."),
        ("Full Curricular Coverage", "Rigorously implemented Units 3, 4, and 5 (PCA, GMM, Markov, HMM, Kalman, CART, Random Forest, AdaBoost, GBDT)."),
        ("Mathematical Grounding", "Every persona decision is derived from first-principles statistical equations from Bishop and Murphy rather than black-box models."),
        ("Interactive Demonstration", "Built a production Streamlit web application with scenario simulation, polar competency radar charts, and downloadable dossiers."),
        ("Reproducible Artifacts", "Codebase fully tested (pytest passed), strictly typed (mypy passed), and synchronized on GitHub.")
    ]
    for c_t, c_d in conclusions:
        p_c = tf14.add_paragraph()
        p_c.text = f"• {c_t}:\n  {c_d}"
        p_c.font.name = "Calibri"
        p_c.font.size = Pt(11.5)
        p_c.font.color.rgb = COLOR_TEXT_MAIN
        p_c.space_before = Pt(7)

    # Right column: References
    rbox14 = slide14.shapes.add_textbox(Inches(6.8), Inches(1.4), Inches(5.6), Inches(5.3))
    rtf14 = rbox14.text_frame
    rtf14.word_wrap = True
    rtf14.margin_left = rtf14.margin_top = rtf14.margin_right = rtf14.margin_bottom = 0

    p = rtf14.paragraphs[0]
    p.text = "Academic References (IEEE Format):"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    refs = [
        "[1] C. M. Bishop, Pattern Recognition and Machine Learning. New York: Springer, 2006. (Bayesian Regression, GMM Expectation-Maximization, Hidden Markov Models).",
        "[2] K. P. Murphy, Machine Learning: A Probabilistic Perspective. Cambridge, MA: MIT Press, 2012. (Kalman Filter State-Space Tracking, Markov Chains).",
        "[3] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, 2nd ed. New York: Springer, 2009. (Decision Trees, AdaBoost, Random Forests).",
        "[4] L. Breiman, 'Random Forests,' Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
        "[5] Y. Freund and R. E. Schapire, 'A Decision-Theoretic Generalization of On-Line Learning,' Journal of Computer and System Sciences, vol. 55, no. 1, 1997.",
        "[6] SRMIST Department of Computing Technologies, Course Curriculum for 21CSC305P Machine Learning, 2026."
    ]
    for r in refs:
        p_r = rtf14.add_paragraph()
        p_r.text = r
        p_r.font.name = "Calibri"
        p_r.font.size = Pt(10)
        p_r.font.color.rgb = COLOR_TEXT_MAIN
        p_r.space_before = Pt(6)

    # Save
    output_pptx = "presentation_unit3_unit4_unit5.pptx"
    prs.save(output_pptx)
    print(f"Generic academic presentation created: {output_pptx} (14 slides, clean standard template).")


if __name__ == "__main__":
    build_generic_presentation()

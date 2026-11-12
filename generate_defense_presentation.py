"""
generate_defense_presentation.py
─────────────────────────────────
Professional, Generic Academic Presentation for SRMIST 21CSC305P Machine Learning.
Topic: Debate-Based Career Guidance Agent with Complementary Personas
Scope: Units 3, 4, 5, and Capstone Multi-Persona Architecture

Key Features:
  - Theory-Focused & Conceptually Rich: Emphasizes mathematical intuition, algorithmic mechanics, and rationale.
  - Minimal Numericals: Eliminates distracting statistics, percentages, and dense decimals.
  - Generic Academic Template: Clean solid navy header bar (#002147), gold divider, crisp white background.
  - Zero Emojis: Clean standard bullet points and professional typography (Calibri).
  - High-Resolution Figures: Real diagrams embedded from figures/ directory.
  - Student Authors: Parth Gupta (RA2411003010019), Mokshith Penna (RA2411003010044), Ishu Gurtu (RA2411003010015).
"""

import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

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


def build_theory_focused_presentation():
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

    def add_slide_picture(slide, default_filename, ppt_filename=None):
        target = None
        if ppt_filename and os.path.exists(ppt_filename):
            target = ppt_filename
        elif os.path.exists(default_filename):
            target = default_filename
        if target:
            pic = slide.shapes.add_picture(target, Inches(6.5), Inches(1.4), width=Inches(6.1))
            pic.line.color.rgb = COLOR_BORDER
            pic.line.width = Pt(1)
            return pic
        return None

    # ── SLIDE 1: Title Slide ──────────────────────────────────────────────────
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
    p_sub.text = "Theoretical Foundations and Algorithmic Synthesis across Units 3, 4, and 5"
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

    # ── SLIDE 2: Conceptual Problem & Theoretical Motivation ──────────────────
    slide2 = add_content_slide(
        "Theoretical Motivation & Multi-Persona Architecture",
        "Overcoming Confirmation Bias and Overconfidence in Automated Advising",
        2
    )
    tbox2 = slide2.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.4), Inches(5.4))
    tf2 = tbox2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = 0

    p = tf2.paragraphs[0]
    p.text = "Limitations of Single-Model Classifiers:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    p_desc = tf2.add_paragraph()
    p_desc.text = "Standard ML classifiers predict a single class label without quantifying whether the student lies on an ambiguous boundary or whether prerequisites are truly met. This leads to overconfident and uncalibrated recommendations."
    p_desc.font.name = "Calibri"
    p_desc.font.size = Pt(12)
    p_desc.font.color.rgb = COLOR_TEXT_MAIN
    p_desc.space_before = Pt(4)

    p_arch = tf2.add_paragraph()
    p_arch.text = "Dialectical Game-Theoretic Solution:"
    p_arch.font.name = "Calibri"
    p_arch.font.size = Pt(14)
    p_arch.font.bold = True
    p_arch.font.color.rgb = COLOR_NAVY_DARK
    p_arch.space_before = Pt(10)

    theory_bullets = [
        ("Thesis (Strategic Planner)", "Optimistic persona that identifies student potential, projects upward mobility, and constructs an ambitious career path."),
        ("Antithesis (Empirical Critic)", "Skeptical persona that cross-examines feasibility against verified alumni skill distributions and flags skill deficits."),
        ("Synthesis (Ethics Sentinel & Arbiter)", "Audits assignment entropy; halts autonomy when ambiguity is high and mediates a balanced consensus recommendation.")
    ]
    for t_title, t_desc in theory_bullets:
        p_t = tf2.add_paragraph()
        p_t.text = f"• {t_title}:\n  {t_desc}"
        p_t.font.name = "Calibri"
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = COLOR_TEXT_MAIN
        p_t.space_before = Pt(6)

    add_slide_picture(slide2, "figures/capstone_debate_architecture.png", "figures/ppt_capstone_debate_architecture.png")

    # ── SLIDE 3: Unit 3 — PCA & Latent Variable Theory ────────────────────────
    slide3 = add_content_slide(
        "Unit 3: Principal Component Analysis (PCA)",
        "Dimensionality Reduction, Orthogonal Subspaces, and Latent Factor Theory",
        3
    )
    tbox3 = slide3.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.4), Inches(5.4))
    tf3 = tbox3.text_frame
    tf3.word_wrap = True
    tf3.margin_left = tf3.margin_top = tf3.margin_right = tf3.margin_bottom = 0

    p = tf3.paragraphs[0]
    p.text = "Theoretical Foundations:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets3 = [
        ("The Multicollinearity Problem", "Academic subjects (e.g., Operating Systems, Programming, and Algorithms) are heavily correlated. Treating them as independent leads to redundant dimensions and distorted distance metrics."),
        ("Orthogonal Coordinate Transformation", "PCA computes the sample covariance matrix and solves the eigen-system Sigma * v = lambda * v. Eigenvectors define uncorrelated axes of maximum variance, while eigenvalues represent variance magnitude."),
        ("The Kaiser Criterion Rationale", "Retains only components with eigenvalues greater than 1.0. Theoretically, a component must explain more variance than a single standardized original variable to be considered informative."),
        ("Latent Competency Extraction", "Condenses high-dimensional academic records into core latent dimensions: Systems Proficiency, Theoretical Logic, and Communication Quotient."),
        ("Practical Role in Guidance", "Allows the Planner persona to map candidates onto an intuitive 2D latent coordinate system without losing core signal.")
    ]
    for b_t, b_d in bullets3:
        p_b = tf3.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    add_slide_picture(slide3, "figures/unit3_pca_scree.png", "figures/ppt_unit3_pca_scree.png")

    # ── SLIDE 4: Unit 3 — GMM & Expectation-Maximization Theory ───────────────
    slide4 = add_content_slide(
        "Unit 3: Gaussian Mixture Models & EM Theory",
        "Soft Responsibilities and Information-Theoretic Uncertainty Quantification",
        4
    )
    tbox4 = slide4.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.4), Inches(5.4))
    tf4 = tbox4.text_frame
    tf4.word_wrap = True
    tf4.margin_left = tf4.margin_top = tf4.margin_right = tf4.margin_bottom = 0

    p = tf4.paragraphs[0]
    p.text = "Why GMM-EM Outperforms K-Means:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets4 = [
        ("Failure of Hard Partitioning (K-Means)", "K-Means enforces rigid 0/1 assignments and assumes spherical clusters. Student competency is inherently continuous—students often exhibit overlapping traits across multiple disciplines."),
        ("Generative Mixture Formulation", "Models the student population as a mixture of K Gaussians, each characterized by its own mean vector and full covariance matrix to capture non-spherical, correlated skills."),
        ("Expectation-Maximization (EM) Principles", "Iteratively optimizes the incomplete log-likelihood: the E-step estimates soft posterior probabilities (responsibilities), and the M-step updates distribution parameters."),
        ("Quantifying Ambiguity with Shannon Entropy", "Entropy H(gamma) measures how evenly spread a student's responsibilities are. Low entropy indicates a clear career archetype; high entropy flags multi-track ambiguity."),
        ("Ethics Sentinel Guardrail", "When entropy exceeds the safety threshold, algorithmic decision-making halts and the candidate is redirected to a human academic counselor.")
    ]
    for b_t, b_d in bullets4:
        p_b = tf4.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    add_slide_picture(slide4, "figures/unit3_gmm_responsibilities_heatmap.png", "figures/ppt_unit3_gmm_responsibilities_heatmap.png")

    # ── SLIDE 5: Unit 3 — Mahalanobis Distance & Outlier Theory ───────────────
    slide5 = add_content_slide(
        "Unit 3: Distance Metrics & Anomaly Detection Theory",
        "Covariance Whitening vs Euclidean Distance for Multivariate Outlier Detection",
        5
    )
    tbox5 = slide5.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.4), Inches(5.4))
    tf5 = tbox5.text_frame
    tf5.word_wrap = True
    tf5.margin_left = tf5.margin_top = tf5.margin_right = tf5.margin_bottom = 0

    p = tf5.paragraphs[0]
    p.text = "Geometric Intuition & Mathematical Rationale:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets5 = [
        ("The Euclidean Flaw in Correlated Data", "Euclidean distance measures straight-line distance assuming all axes are independent. It treats a drop in correlated subjects (e.g. OS and Algorithms) as two independent penalties, exaggerating true divergence."),
        ("Mahalanobis Covariance Whitening", "Incorporates the inverse covariance matrix Sigma^-1. Geometrically rotates into the principal axes and scales by eigenvalue variances, measuring distance in standard deviations from the cluster centroid."),
        ("Statistical Anomaly Detection", "Detects atypical competency shapes. For example, a student with high communication but near-zero programming diverges strongly from software engineering norms, even if their overall average seems acceptable."),
        ("Empirical Critic Integration", "The Critic uses Mahalanobis distance as an objective feasibility gate: candidates lying too far from successful alumni centroids are flagged for prerequisite remediation.")
    ]
    for b_t, b_d in bullets5:
        p_b = tf5.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(9)

    add_slide_picture(slide5, "figures/unit3_clustering_comparison.png", "figures/ppt_unit3_clustering_comparison.png")

    # ── SLIDE 6: Unit 4 — Markov Chains & Transition Dynamics ────────────────
    slide6 = add_content_slide(
        "Unit 4: Sequential Career Modeling with Markov Chains",
        "Stochastic State Transitions, Chapman-Kolmogorov Dynamics, and Equilibrium",
        6
    )
    tbox6 = slide6.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.4), Inches(5.4))
    tf6 = tbox6.text_frame
    tf6.word_wrap = True
    tf6.margin_left = tf6.margin_top = tf6.margin_right = tf6.margin_bottom = 0

    p = tf6.paragraphs[0]
    p.text = "Stochastic Career Modeling Principles:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets6 = [
        ("Career as a Stochastic Process", "Career mobility is not static; it is a sequence of transitions across discrete professional tiers: Junior Associate, Mid-Level Engineer, Senior Specialist, Lead Architect, and Executive."),
        ("The First-Order Markov Property", "Assumes that the probability of transitioning to the next career stage depends strictly on the current state, encapsulating career momentum without historical baggage."),
        ("Multi-Step Prediction via Matrix Powers", "By Chapman-Kolmogorov equations, the k-step transition matrix A^k yields long-term forecasts of where candidates will be after multiple evaluation cycles."),
        ("Stationary Distribution & Absorbing States", "Solving pi* A = pi* reveals the long-term equilibrium distribution, ensuring the model possesses realistic career absorption rather than infinite cycling."),
        ("Career Plateau & Bottleneck Detection", "Identifies transition bottlenecks (e.g., the mid-career plateau where promotion stalls), allowing the system to recommend targeted skill interventions early.")
    ]
    for b_t, b_d in bullets6:
        p_b = tf6.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    add_slide_picture(slide6, "figures/unit4_markov_transition_dynamics.png", "figures/ppt_unit4_markov_transition_dynamics.png")

    # ── SLIDE 7: Unit 4 — Hidden Markov Models & Viterbi Decoding ─────────────
    slide7 = add_content_slide(
        "Unit 4: Hidden Markov Models & Viterbi Decoding",
        "Decoupling Hidden Competency Maturity from Noisy Academic Observations",
        7
    )
    tbox7 = slide7.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.4), Inches(5.4))
    tf7 = tbox7.text_frame
    tf7.word_wrap = True
    tf7.margin_left = tf7.margin_top = tf7.margin_right = tf7.margin_bottom = 0

    p = tf7.paragraphs[0]
    p.text = "Latent State Formulation & Dynamic Programming:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets7 = [
        ("The Latent Competency Concept", "True student mastery (Novice, Competent, Expert) is unobservable directly. What we measure are noisy indicators: semester marks, hackathons, and project assessments."),
        ("The Dual Stochastic Process", "An underlying hidden Markov process dictates student maturity progression, while an emission probability distribution governs the observable grades generated at each stage."),
        ("Evaluation via Scaled Forward-Backward", "Calculates sequence likelihood while utilizing dynamic scale factors to avoid numerical underflow across long academic sequences."),
        ("Viterbi Maximum A Posteriori (MAP) Decoding", "Utilizes Bellman's Principle of Optimality to find the single most probable trajectory of hidden states without evaluating exponentially many paths."),
        ("Actionable Roadmap Generation", "Produces an interpretable progression path (e.g. Novice -> Competent -> Expert) that the Strategic Planner translates into semester milestones.")
    ]
    for b_t, b_d in bullets7:
        p_b = tf7.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    add_slide_picture(slide7, "figures/unit4_workflow.png", "figures/ppt_unit4_workflow.png")

    # ── SLIDE 8: Unit 4 — Kalman Filtering for Skill Tracking ─────────────────
    slide8 = add_content_slide(
        "Unit 4: Linear Dynamical Systems & Kalman Filter",
        "Continuous Capability Tracking and Recursive State-Space Filtering",
        8
    )
    tbox8 = slide8.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.4), Inches(5.4))
    tf8 = tbox8.text_frame
    tf8.word_wrap = True
    tf8.margin_left = tf8.margin_top = tf8.margin_right = tf8.margin_bottom = 0

    p = tf8.paragraphs[0]
    p.text = "Continuous State-Space Estimation Theory:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets8 = [
        ("Beyond Discrete Stages", "Student capability is not merely discrete states; it evolves continuously with a true underlying proficiency level and a growth rate (velocity)."),
        ("State-Space Modeling Framework", "Formulates system dynamics: state transition projects future capability with process uncertainty Q; measurement matrix maps true capability to noisy exam scores with review noise R."),
        ("The Predict-Update Operating Cycle", "Prediction propagates state ahead in time; Measurement Update uses new exam scores to correct predictions via the mathematically optimal Kalman Gain."),
        ("Optimal LMMSE Estimator", "Under Gaussian assumptions, the Kalman filter is proven to be the optimal Linear Minimum Mean Squared Error estimator, balancing prior model confidence against measurement noise."),
        ("Filtering Academic Transients", "Smooths out temporary dips caused by exam anxiety or illness, revealing the true underlying capability growth trajectory.")
    ]
    for b_t, b_d in bullets8:
        p_b = tf8.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    add_slide_picture(slide8, "figures/unit4_kalman_filter.png", "figures/ppt_unit4_kalman_filter.png")

    # ── SLIDE 9: Unit 5 — Decision Trees & Pruning Theory ────────────────────
    slide9 = add_content_slide(
        "Unit 5: Decision Trees & Regularization Theory",
        "Recursive Space Partitioning, Impurity Criteria, and the Bias-Variance Tradeoff",
        9
    )
    tbox9 = slide9.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.4), Inches(5.4))
    tf9 = tbox9.text_frame
    tf9.word_wrap = True
    tf9.margin_left = tf9.margin_top = tf9.margin_right = tf9.margin_bottom = 0

    p = tf9.paragraphs[0]
    p.text = "CART Splitting & Regularization Principles:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets9 = [
        ("Recursive Binary Partitioning", "CART partitions the multidimensional feature space into orthogonal rectangular regions using greedy, top-down recursive binary splits."),
        ("Gini Impurity vs Information Gain", "Gini measures the expected error rate if instances were randomly labeled according to node class distribution. Minimizing Gini isolates pure, single-track subsets."),
        ("The High Variance Problem", "Unconstrained decision trees grow deep branches to fit every training anomaly, achieving near-zero training error but high test variance."),
        ("Minimal Cost-Complexity Pruning", "Introduces an explicit complexity penalty alpha * |T| penalizing tree size. Balances misclassification loss against leaf count to prune subtrees that fail to generalize."),
        ("Interpretability Advantage", "Produces transparent, rule-based decision logic that human counselors and students can inspect, audit, and understand.")
    ]
    for b_t, b_d in bullets9:
        p_b = tf9.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    add_slide_picture(slide9, "figures/unit5_workflow.png", "figures/ppt_unit5_workflow.png")

    # ── SLIDE 10: Unit 5 — Bagging vs Boosting (RF vs AdaBoost) ──────────────
    slide10 = add_content_slide(
        "Unit 5: Bagging vs Boosting Paradigms",
        "Variance Reduction via Independent Trees vs Sequential Bias Reduction",
        10
    )
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

    rf_pts = [
        "Core Philosophy: Combines many deep, low-bias, high-variance trees independently to drive variance down.",
        "Bootstrap Sampling: Each tree trains on a random subset with replacement, exposing each model to different data variations.",
        "Feature Subspacing (sqrt(D)): Randomly restricts candidate features at each node split, decorrelating individual tree errors so errors cancel out upon averaging.",
        "Out-of-Bag (OOB) Generalization: Unused instances serve as an internal validation set, providing unbiased error estimates without data leakage.",
        "Practical Behavior: Highly robust to label noise, feature correlation, and outlier profiles."
    ]
    for pt in rf_pts:
        p = btf1.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p.space_before = Pt(8)

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

    ada_pts = [
        "Core Philosophy: Sequentially trains simple, shallow, high-bias stumps to progressively reduce systematic bias.",
        "Iterative Sample Reweighting: Instances misclassified in round m receive higher weights in round m+1, forcing subsequent models to focus on hard cases.",
        "Exponential Loss Minimization: Formulates boosting as coordinate descent minimizing an upper bound on classification error.",
        "Model Aggregation: Assigns higher voting weights to accurate classifiers and lower weights to weak ones.",
        "Practical Vulnerability: In noisy educational records, ambiguous or mislabeled border cases cause sample weights to explode, degrading generalization."
    ]
    for pt in ada_pts:
        p = btf2.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p.space_before = Pt(8)

    # ── SLIDE 11: Unit 5 — Gradient Boosted Trees (GBDT Theory) ──────────────
    slide11 = add_content_slide(
        "Unit 5: Gradient Boosted Trees & Model Benchmarking",
        "Functional Gradient Descent in Function Space and Shrinkage Regularization",
        11
    )
    tbox11 = slide11.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.4), Inches(5.4))
    tf11 = tbox11.text_frame
    tf11.word_wrap = True
    tf11.margin_left = tf11.margin_top = tf11.margin_right = tf11.margin_bottom = 0

    p = tf11.paragraphs[0]
    p.text = "Functional Gradient Descent Theory:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    bullets11 = [
        ("Optimization in Function Space", "Unlike standard gradient descent which optimizes parameter weights, GBDT optimizes entire decision functions by stepping along the negative gradient of the loss surface."),
        ("Fitting Pseudo-Residuals", "Each new tree is trained directly on the pseudo-residuals (unexplained errors) of previous trees, iteratively refining decision boundaries."),
        ("Loss Function Generalization", "Applicable to any differentiable loss function (e.g. multi-class cross-entropy deviance), providing calibrated probability outputs."),
        ("Shrinkage Regularization (Learning Rate)", "Dampens each tree's step size with a shrinkage factor, requiring more trees but drastically preventing greedy overshooting."),
        ("Architectural Selection", "Random Forest provided the most stable generalization across career tracks, while GBDT offered valuable probability estimates for debate arbitration.")
    ]
    for b_t, b_d in bullets11:
        p_b = tf11.add_paragraph()
        p_b.text = f"• {b_t}:\n  {b_d}"
        p_b.font.name = "Calibri"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MAIN
        p_b.space_before = Pt(8)

    add_slide_picture(slide11, "figures/unit5_ensemble_comparison.png", "figures/ppt_unit5_ensemble_comparison.png")

    # ── SLIDE 12: Capstone — Dialectical Multi-Persona Architecture ───────────
    slide12 = add_content_slide(
        "Integrated Capstone: Multi-Persona Guidance Architecture",
        "Adversarial Personas with Mathematical Safety Guardrails",
        12
    )
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
    p.text = "Strategic Career Planner\n(Growth & Ambition)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK
    pts1 = [
        "Theoretical Role: Acts as the student's champion; maximizes career potential and earnings.",
        "Bayesian Prediction: Utilizes analytical Bayesian regression to project salary with credible intervals.",
        "Viterbi Progression: Decodes optimal multi-semester milestones to reach the target tier.",
        "Guiding Philosophy: Ambitious upward mobility based on latent capabilities."
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
    p.text = "Empirical Critic\n(Feasibility & Metric Audit)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_GOLD
    pts2 = [
        "Theoretical Role: Acts as the reality check; minimizes risk of academic failure or burnout.",
        "Mahalanobis Gate: Checks multivariate distance against verified alumni distributions.",
        "Transition Audit: Examines Markov retention plateaus to spot stagnation risks early.",
        "Guiding Philosophy: Strict feasibility verification and prerequisite enforcement."
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
        "Theoretical Role: Guarantees algorithmic safety and fairness.",
        "Shannon Entropy Audit: Measures classification ambiguity across cluster responsibilities.",
        "Human Counselor Fallback: Halts autonomous AI advisement when entropy is high.",
        "Guiding Philosophy: Never allow an algorithm to make blind guesses on student futures."
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
    p.text = "Arbiter Consensus Adjudication"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(22, 163, 74)

    p_a = atf.add_paragraph()
    p_a.text = "Acts as the impartial judge that balances ambition against feasibility and safety. Weighs all three persona arguments to deliver a calibrated confidence score, final milestone recommendation, and customized 4-semester skill development roadmap."
    p_a.font.name = "Calibri"
    p_a.font.size = Pt(11)
    p_a.font.color.rgb = COLOR_TEXT_MAIN
    p_a.space_before = Pt(3)

    # ── SLIDE 13: Capstone — Scenario Evaluation & Guardrails ─────────────────
    slide13 = add_content_slide(
        "Capstone: Case Studies & Guardrail Verification",
        "Empirical Validation of Dialectical Personas across Representative Archetypes",
        13
    )
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
         "Conceptual Dynamics: Strong core competencies in Algorithms, OS, and Coding.\n• Planner: Recommends accelerated AI / Machine Learning Specialist track.\n• Critic: Confirms profile lies firmly within alumni distribution.\n• Sentinel: Confirms sharp, unambiguous classification.\n• Verdict: Unanimous Dialectical Consensus with maximum confidence.",
         COLOR_NAVY_DARK),

        ("Case Study 2: Skill Deficit Anomaly",
         "Conceptual Dynamics: High interest in AI, but foundational Math and Algorithm scores lag.\n• Planner: Identifies target milestone role.\n• Critic: Flags severe Mahalanobis distance outlier anomaly.\n• Verdict: Conditional Advancement. Student must complete 2 semesters of prerequisite remedial bootcamps before specialization.",
         COLOR_GOLD),

        ("Case Study 3: Ambiguous Boundary Profile",
         "Conceptual Dynamics: Uniformly moderate performance across all competency areas.\n• Planner & Critic: Unable to decisively separate tracks.\n• Sentinel: Detects high Shannon entropy across multiple clusters.\n• Verdict: Human Counselor Override. Autonomous guidance is halted; candidate is escalated to an Academic Counselor.",
         RGBColor(220, 38, 38)),

        ("Case Study 4: Cloud & Systems Architect",
         "Conceptual Dynamics: High aptitude in Computer Networks, Systems, and Architecture.\n• All Personas: Unanimous agreement on Enterprise Infrastructure Engineer.\n• Verdict: Full consensus with specialized cloud deployment milestones.",
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
    tbox14 = slide14.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tf14 = tbox14.text_frame
    tf14.word_wrap = True
    tf14.margin_left = tf14.margin_top = tf14.margin_right = tf14.margin_bottom = 0

    p = tf14.paragraphs[0]
    p.text = "Core Theoretical Contributions:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    conclusions = [
        ("Multi-Agent Dialectical Robustness", "Adversarial cross-examination replaces brittle single-model overconfidence with explainable, balanced career guidance."),
        ("Comprehensive Curricular Unification", "Successfully harmonized Unsupervised Learning (Unit 3), Sequential Modeling (Unit 4), and Ensemble Methods (Unit 5)."),
        ("Mathematical Auditability", "Every recommendation is grounded in first-principles statistical equations rather than opaque, uncalibrated black boxes."),
        ("Safety & Human-in-the-Loop", "Entropy-triggered counselor overrides ensure automated systems recognize their own uncertainty and protect student welfare."),
        ("Reproducible Open-Source Delivery", "Delivered complete codebase with unit test validation, executed notebooks, and an interactive demonstration platform.")
    ]
    for c_t, c_d in conclusions:
        p_c = tf14.add_paragraph()
        p_c.text = f"• {c_t}:\n  {c_d}"
        p_c.font.name = "Calibri"
        p_c.font.size = Pt(11.5)
        p_c.font.color.rgb = COLOR_TEXT_MAIN
        p_c.space_before = Pt(7)

    rbox14 = slide14.shapes.add_textbox(Inches(6.8), Inches(1.4), Inches(5.6), Inches(5.3))
    rtf14 = rbox14.text_frame
    rtf14.word_wrap = True
    rtf14.margin_left = rtf14.margin_top = rtf14.margin_right = rtf14.margin_bottom = 0

    p = rtf14.paragraphs[0]
    p.text = "Primary Theoretical References (IEEE Format):"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_DARK

    refs = [
        "[1] C. M. Bishop, Pattern Recognition and Machine Learning. New York: Springer, 2006. (Bayesian Linear Regression, GMM Expectation-Maximization, Hidden Markov Models).",
        "[2] K. P. Murphy, Machine Learning: A Probabilistic Perspective. Cambridge, MA: MIT Press, 2012. (Kalman Filter State-Space Tracking, Markov Chains, Ensemble Methods).",
        "[3] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, 2nd ed. New York: Springer, 2009. (Decision Trees, AdaBoost, Random Forests, GBDT).",
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
    print(f"Theory-focused generic presentation created: {output_pptx} (14 slides).")


if __name__ == "__main__":
    build_theory_focused_presentation()

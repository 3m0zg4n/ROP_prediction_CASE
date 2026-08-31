# MULTI-CRITERIA DECISION-MAKING IN PREDICTING DRILLING SPEED
## A Holistic Approach to Rate of Penetration (ROP) Optimization

**Document ID:** MCDM-ROP-2026-001  
**Date:** January 12, 2026  
**Project:** ROP Prediction System (CBT-LSTM)  
**Reference Documents:** 
- CASE_STUDY_CBT_LSTM.md
- rop_prediction_system.py
- PREDICTION_GAP_EXPLANATION.md
- DIAGRAM_INTERPRETATION_GUIDE.md

---

## 1. EXECUTIVE SUMMARY

Drilling optimization is fundamentally a complex Multi-Criteria Decision-Making (MCDM) problem. The objective—maximizing Rate of Penetration (ROP)—is not a simple linear function of input power. It is a balancing act constrained by mechanical limits, hydraulic efficiency, and geological unpredictability.

This document outlines the multi-criteria framework used in the intelligent ROP prediction system. By moving beyond simple regression and adopting a "Multi-Criteria Thinking Paradigm," the system integrates **Physics**, **Geology**, and **Time-Series Dynamics** to provide accurate, actionable drilling speed predictions. This approach moves the focus from "fitting curves" to "understanding the drilling process."

---

## 2. THE MULTI-CRITERIA FRAMEWORK

The system's architecture is built upon three distinct criteria pillars, mimicking the cognitive process of a human expert driller who weighs multiple factors simultaneously.

### The "Three-Pillar" Logic Model

| Criterion Pillar | Focus Area | Question Answered | Key Inputs |
| :--- | :--- | :--- | :--- |
| **Criterion A: Operational Control** | Surface Parameters | *What are we doing to the bit?* | WOB, RPM, Torque, Flow, SPP |
| **Criterion B: Environmental Context** | Geology & Formation | *What are we drilling through?* | Gamma Ray, Sonic, Density, Resistivity |
| **Criterion C: Physical Constraints** | Drilling Physics | *Is the process efficient?* | MSE, HSI, UCS, Stick-Slip |

---

## 3. CRITERION 1: OPERATIONAL PARAMETERS (The "Control" Factor)

These are the decision variables available to the driller or the automated control system. They represent the energy input into the system.

### Key Decision Variables
1.  **Weight on Bit (WOB):** The primary force driving the bit into the rock.
    *   *Decision Impact:* Higher WOB increases ROP up to the "Founder Point," after which it causes bit balling or vibration.
2.  **Rotation Speed (RPM):** The speed at which the bit turns.
    *   *Decision Impact:* Higher RPM improves cutting cleaning and speed but increases heat and wear.
3.  **Flow Rate (GPM):** The volume of drilling fluid pumped downhole.
    *   *Decision Impact:* Critical for hole cleaning (removing cuttings). Insufficient flow causes "re-drilling" of cuttings, plummeting ROP.
4.  **Standpipe Pressure (SPP):** The pressure required to circulate fluid.
    *   *Decision Impact:* An indicator of downhole restrictions or efficient nozzle selection.
5.  **Surface Torque:** The rotational force measured at the rig floor.
    *   *Decision Impact:* High fluctuations indicate stick-slip or hole issues.

### Visualizing Parameter Influence
To verify that the model correctly interprets these controls, we utilize the **Parameter Influence Scatter Plot** (`Parameter_Influence_Scatter.png`).
*   **Purpose:** This acts as a "sanity check" to ensure the AI hasn't learned non-physical correlations.
*   **Interpretation:** We compare the "Actual Physics" (Left Column) vs. "Model Physics" (Right Column). A valid model will replicate the physical relationship—e.g., as WOB increases, ROP typically increases (linear positive correlation) until the Founder Point is reached.


---

## 4. CRITERION 2: GEOLOGICAL CONTEXT (The "Environment" Factor)

A decision that is correct in soft sandstone (e.g., high RPM) might be catastrophic in hard abrasive quartzite. The system uses "Formation Evaluation" data to adjust its expectations.

### Geological Inputs
1.  **Gamma Ray (GR):** Measures natural radioactivity.
    *   *Role:* Distinguishes between reactive clays (Shale) and reservoir rocks (Sand/Limestone).
    *   *Derived Feature:* **V_SH (Volume of Shale)** - Quantifies rock "clays."
2.  **Sonic Logs (DT):** Measures sound speed through rock.
    *   *Role:* A direct proxy for rock hardness and porosity.
    *   *Derived Feature:* **UCS (Unconfined Compressive Strength)** - Estimates the crushing strength of the rock in MPa.
3.  **Density (RHOB) & Porosity:**
    *   *Role:* Identifies how dense or porous the formation is, affecting drillability.

**Gap Analysis Insight:** As noted in `PREDICTION_GAP_EXPLANATION.md`, the "Geological Domain Shift" (moving from soft to hard rock) is the #1 cause of prediction error. A robust MCDM system must weigh this criterion heavily to avoid over-predicting ROP in harder formations.

### Visualizing Geological Response
The model's responsiveness to this criterion is best observed in the **Depth Profile Charts** (e.g., `Depth_Plot_Well_X_1.png`).
*   **The View:** This places the Predicted ROP (Blue Line) against the Actual ROP (Black Line) along the wellbore depth.
*   **The Evidence:** In Well X-1, we observe the model tracking sharp ROP changes that correlate perfectly with lithology boundaries. This "tight tracking" (R² = 0.834) confirms the model is actively "listening" to the Gamma Ray and Sonic logs to make its decisions, rather than just averaging the inputs.


---

## 5. CRITERION 3: PHYSICS-INFORMED CONSTRAINTS (The "Law" Factor)

"Data without Physics is Hallucination." This criterion acts as a sanity check, ensuring predictions adhere to physical laws.

### Physics Pillars
1.  **Mechanical Specific Energy (MSE):**
    *   *Formula:* $MSE = \frac{WOB}{A_b} + \frac{120 \pi \cdot RPM \cdot Torque}{A_b \cdot ROP}$
    *   *The Decision:* If MSE is rising while ROP is flat, the system detects **inefficiency** (dysfunction). The model learns that simply adding more WOB will not result in higher ROP.
2.  **Hydraulic Specific Energy (HSI):**
    *   *Function:* Measures the hydraulic horsepower delivered to the bit face.
    *   *The Decision:* Ensures sufficient cleaning energy is present to support the predicted ROP.

---

## 6. THE DECISION ENGINE: CBT-LSTM ARCHITECTURE

The **Channel Boosted Time-Series Long Short-Term Memory (CBT-LSTM)** network acts as the "Decision Engine," integrating these valid but often conflicting criteria.

### How It Synthesizes Decisions:
1.  **Temporal Awareness (Memory):**
    *   Unlike static equations (Bourgoyne & Young), the LSTM remembers the "state" of the well. It knows that high Torque *now* is a result of the lithology change *5 minutes ago*.
### Verifying Model Bias
To ensure the "Decision Engine" is impartial (not systematically over- or under-predicting), we employ two specific visualizations:
*   **Global Parity Plot (`Parity_Plot_Global.png`):**
    *   *What it shows:* A direct comparison of decisions vs. reality.
    *   *Success Metric:* The tight clustering of points along the red 1:1 diagonal proves the model weighs criteria effectively across different depth ranges (represented by color).
*   **Residual Analysis (`Residual_Distribution.png`):**
    *   *What it shows:* The distribution of prediction errors.
    *   *Success Metric:* The "Bell Curve" shape centered at zero confirms that errors are random noise, not a flaw in the decision logic. A centered peak means the Multi-Criteria integration is balanced.

2.  **Multi-Scale Processing (Channel Boosting):**
    *   It separates "Fast" criteria (vibration, shock) from "Slow" criteria (bit wear, formation change).
    *   *Decision:* It filters out high-frequency noise (which shouldn't drive decision-making) while retaining long-term trends.
3.  **Optimization:**
    *   As detailed in `OPTIMIZATION_REPORT.md`, the model is tuned (Layers, Units, Dropout) to minimize Mean Absolute Error (MAE), finding the optimal balance between these inputs.

---

## 7. CASE STUDY OUTCOMES & INTERPRETATION

The application of this Multi-Criteria thinking was validated in `CASE_STUDY_CBT_LSTM.md`.

### Performance Hierarchy
1.  **Baseline (Well X-1):** High accuracy (R² ~0.83) in known conditions.
2.  **Generalization (Well X-2, X-3):** Maintained robust prediction (R² ~0.78 - 0.80) even with different operational parameters (Criterion A change).
3.  **The "Gap" (Domain Shift):** The system successfully identified that disparate ROP values were due to geological changes (Criterion B), not model failure.

### Visualizing the Decision
As per the `DIAGRAM_INTERPRETATION_GUIDE.md`:
*   **Depth Plots:** Show the continuous decision/prediction path.
*   **Parity Plots:** Visualize the global adherence to the "Multi-Criteria Truth."
*   **Residual Analysis:** Confirms no systematic bias exists in the decision logic.

---

## 8. CONCLUSION

The effective prediction of drilling speed is not merely a statistical exercise; it is a multi-dimensional engineering problem. By explicitly modeling the **Operational Controls**, **Geological Environment**, and **Physical Constraints**, the CBT-LSTM system moves beyond black-box AI. It provides a transparent, physics-compliant decision support tool capable of navigating the "Drilling Optimization Paradox."

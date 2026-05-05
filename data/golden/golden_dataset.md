# MedRAG-Agent Golden Dataset

> 填写说明：
> - 每道题用 `## Qxxx` 开头（三位数字，如 Q001）
> - **Category** 从以下选：Pharmacology / Oncology / Radiology / Cardiology / Neurology / Infectious Disease / General
> - **Difficulty** 填：Easy / Medium / Hard
> - **Question** 填具体问题（英文，面向文献的事实性问题）
> - **Answer** 填期望答案（完整句子，可多段）
> - **Notes**（可选）：填写出处提示、特殊说明等
> - 题目之间用 `---` 分隔

---

## Q001

**Category**: Radiology
**Difficulty**: Easy

**Question**: What is the typical spatial resolution of 3T MRI in clinical practice?

**Answer**: The typical spatial resolution of 3T MRI in clinical practice is approximately 0.67 mm × 0.67 mm in-plane with a slice thickness of around 3–3.5 mm. Standard protocols commonly use a matrix size of 256 × 256, a field of view of about 160–240 mm, and a slice thickness of approximately 3 mm.

**Notes**: Verified against PMC corpus. Good for testing P1 vs P3 ranking quality.

---

## Q002

**Category**: Pharmacology
**Difficulty**: Medium

**Question**: What is the primary mechanism of action of metformin in treating type 2 diabetes?

**Answer**: Metformin primarily acts by inhibiting hepatic gluconeogenesis through activation of AMP-activated protein kinase (AMPK). It reduces glucose production in the liver, decreases intestinal absorption of glucose, and improves insulin sensitivity in peripheral tissues. Unlike sulfonylureas, it does not stimulate insulin secretion.

**Notes**: Out-of-domain for current PubMed corpus — useful for testing "refusal to answer" behavior.

---

## Q003

**Category**: Oncology
**Difficulty**: Medium

**Question**: What is the role of BRCA1 in DNA damage repair?

**Answer**: BRCA1 plays a central role in the homologous recombination (HR) pathway of DNA double-strand break repair. It forms a complex with BRCA2 and RAD51 to facilitate error-free repair of DNA lesions. BRCA1 also participates in cell cycle checkpoint activation, ensuring cells do not divide before DNA damage is repaired. Germline mutations in BRCA1 significantly increase lifetime risk of breast and ovarian cancers.

**Notes**: Partially covered in PubMed abstracts. Tests whether sparse retrieval improves recall of gene-name exact matches.

---

## Q004

**Category**: 
**Difficulty**: 

**Question**: 

**Answer**: 

**Notes**: 

---

## Q005

**Category**: 
**Difficulty**: 

**Question**: 

**Answer**: 

**Notes**: 

---

<!-- 继续填写 Q006 到 Q050，格式与上方相同 -->

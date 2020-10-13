# Bioinformatics
- interdisciplinary, understading biological data, using mathematical and statistical techniques 

**Goals**
- Sequence alignment
- Gene finding
- Drug discovery and drug design
- Protein structure alignment and predicton 

**Current State**
- 

**Research Opportunities**
- 

## Tools and ML/AI
**Python**
- [`Biopython`](https://biopython.org/): py lib for biological computation and seq analysis 
- 



# DNA Sequencing
- To understand features, function, structure or evolution of DNA or RNA 
- A sequence where order matters. What can we transfer from NLP and TimeSeries analysis? 
- (Chap 4: Principles and Methods of Seq Analysis, NCIB)[https://www.ncbi.nlm.nih.gov/books/NBK20261/?report=reader]

**Scope**
- DNA sequencing: order of nucleotides 
- RNA sequencing: number of RNA in sample 
- Protein sequencing: amino acid order 

**Metrics**
- 


**Approaches**
- Targeted sequencing: 
- 
**Methods**
- NGS
- Sanger
- qPCR
- Micro-array

**Some Context and Basics**
- DNA
    - is same in nearly every cell of our bodies
    - Mostly located in cell nucleus but traces possible in mitochondria (mt) and so mtDNA. 
    - Structure: 4 chemical bases (AGCT). 3B+ bases in humans and 99% of them are similar in all humans. << What's up with the 1%?  Order matters. Nucleotide = sugar + base + phosphate. Double stranded, helix form 
    - Coding DNA (1%): protein making instructions. 
        - may contain some nonCoding DNA regions (introns). removed before protein is made
        - Introns: can contain regulators 
    - NonCoding DNA (99%): Control of gene activity - regulatory elements provide transcription factors proteins (binding spots) for transcription (on/off genes). << control = trigger, hinder, enhance, >>
        - Regulators of transcription 
            - promoters: transcription machinery. prefix a gene 
            - enhancers: activate transcription. can be anywhere
            - silencers: repress transctiption. can be anywhere
            - insulators: control transcription by blocking enhancers, hinder (barrier to) dna structural changes that repress gene activity 
        - RNA molecule formation instructions - regulate (but not code for) protein formation?? e.g for 
            - transfer RNAs (tRNA): 
            - ribosomal (rRNA): help in protein assembly by ??
            - micro (miRNA): block protein production process
            - long noncoding (lncRNA): diverse roles in regulating gene activity << diff with regulators?? >>
        - structural part of chromosome
            - telomeres: repeated, at ends of chromosomes. Protect end of chromes from degradation during copying
            - satellite DNA: repetitive, basis of centromere (kink in pairs) and heterochromatin (controlling gene activity, maintaining chromosome structure)
        - noncoding DNA regions in other stuctures 
            - Introns: within coding DNA << avail regulators?? >>
            - Intergenetic regions: between genes

    - Replicates

- RNA

- Gene
    - basic physical and functional unit of heredity; atom of heredity
    - Structure: Gene = subsets of DNA fragments??. 100s to 2M+ DNA bases. 20K-25K genes in humans and 2 copies each (X-Y chromosome sources). 99% genes same for all humans. 
        - Allele = variants of same gene; small diffs in sequence. -> diffs in physical features
    - Functions
        - instruction to make proteins (but most genes don't do this - so what are they doing? what implications @ domancy)
- Chromosome
    - packed/tightly-coiled structure of DNA. = DNA + histones protein (structural). In nuclues (so no in mitochondria?? )
    - Not visible when the cell is not dividing b/c more tightly packed during cell division 
    - Centromere (the kink) used to describe location of specific genes. << How else to describe gene location? >>
    - 23 pairs in humans. 22 pairs are autosomes and 'gender-agnostic'. 1 pair (# 23) is the sex chromosome. 
        - Autosomes are number in descending order of size
        - What's the role of each pair? Why different sizes and shapes?     
    ![Chromosome structure](https://ghr.nlm.nih.gov/primer/illustrations/chromosomestructure.jpg)


- Phenotype


---- 

# Next Generation Sequencing (NGS)
- Price:
- Migration/Interoperabilit: migration support for qPCR
- (link)[https://www.illumina.com/science/technology/next-generation-sequencing/beginners.html]


**Vs Sanger Seq**
- Sanger = dideoxy or capillary electrohoresis sequencing
- Similarities
    - DNA polymerase adds fluorescent nucleotides 
    - each nucleotide is identified by is fluorescent tag
    - discovery??? 
- Differences = parallelization (throughput) + discovery poer. 
    - sangers efficient if few targets (<=20); NGS is expensive at that level
    - Sangers sequences one DNA fragment at a time. NGS does millions of fragments concurrently (parallel )
    - NGS higher ability to detect novel or rare variants via deep sequencing 
**Vs qPCR**
- Similarities
    - high sensitivity
    - reliable variant detection 

- Differences - priori and parellization (throughput)  
    - qPCR can only detect known sequences. No discovery power
    - NGS has higher discovery power (genes) and sensitivity (rare variants and transcripts)
    - qPCR good for low target numbers (<=20). NGS is expensive at that level. NGS parellize multiple targets and samples 

**Vs MicroArrays**
- Similarities
    - 
- Differences (NGS here is eq to gen RNA-seq)
    - discovery and detection of rare transcripts; mciroarray doesn't b/c it needs specific probes of the species/transcript in question 
    - wider dynamic ranage of RNA-sequencing
    - higher specificity and sensitivity of sequencing
    - 

**Terms**
- Hypothesis-free experimental design. No prior knowledge required. << affects: bias, discovery ability 
- qPCR: 
    - Suitable for few targets, screening/identification purpose, 



## Example: (Gen Guide + MultiQC, Dr. J. Q. Oliete)[https://www.kolabtree.com/blog/a-step-by-step-guide-to-dna-sequencing-data-analysis/]
-  

## Example: (Using Biopython, Adnan)[http://blog.adnansiddiqi.me/python-for-bioinformatics-getting-started-with-sequence-analysis-in-python/]
[MIT Open Course Ware](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-096-algorithms-for-computational-biology-spring-2005/index.htm)

https://www.youtube.com/watch?v=lJzybEXmIj0&list=PLUl4u3cNGP63uK-oWiLgO7LLJV6ZCWXac&index=1


# Chap 1: Intro 
**History**
- 70s: similarity matrices and molecular evolution (3rd group Archaea, molecular clock idea Vs phenotype? )
- 80s: seq db growth, seq search and aligment algs e.g. BLAST, FastA 
- 90s: seq labeling,identification, Hidden Markov Models, predicting protein structure from seq 
- 2000s: genome sequencing growth, larger organisms genome dbs, high throughput methods/tools - massively parallel, metgenomics, bioimage informatics,  birth of systems biology and synthetic biology & biological engineering, 
- 2010s: NextGenSequencing (NGS), transcriptome seq is now routine, protein-D/RNA interactions, genome sequencing

Topics: genomic analysis, modeling biological function, proteomics, regulatory networks, computational genetics


**RQs** 
Dig us: what information is there, can we control it and in some given ways, can we predict/pre-empt, evolutionary history information 
Health\Evolve: indicators and measurement, therapy design, individual Vs systemic, genome role, 
- what's encoded in genome
- how are chromosomes organize? what genes are present
- what regulatory circutry is there/encoded
- can you predict transcriptome from the genome and proteome from transcriptome
- can protein function be predicted from sequence
- can evolutionary history be reconstructed from sequence
- what to measure to discover cause of disease, drug mechanism, metabolic pathways
- what kind of modeling to desing new therapies or reengineer 
- what can we currently measure, meaning of each data individually and as a system 
- computation @ big data 

## Computational Challenges
- big data handling + tech advancements 
- efficient indexing of sequences - alignment, searching, repetitive elements handling
- parallized sequencing, high-throughput 
- gene expressiong - mapping RNA back to DNA
- outside D/RNA activity - protein interactions and structures, from gene to protein and reverse predictions
- protein-dna interaction - gene regulatory networks 
- computable models - ;bayesian networks




------------------
# Chap 2: Local Alignment (BLAST) and Statistics
- Classical sequencing Vs NGS analytical methods
- BLAST 

## Sequencing
- is usually on DNA; RNA gets converted to DNA first 
- Chemistry of seqyuencing is related to the type of nucleotide 
	- ribo
	- deoxyrib
	- dideoxyrib

- Molecular structure:: TODO: review 
	- numbering 1 to 5 
	- 5 primer: 5 connects to the phosphate and so .... 
	- 3 primer: extend at 3: if no OH at 3 then can't extend
- Sequence order:
	- 5 primer to 3 primer direction 
	
**Sanger Sequencing Method**
- origin. Takes advantage of the fact that the dideoxy terminates the growing chain
- 


setwd("~/3.TCGA/TCGA-Assembler-2/TCGA-Assembler")
source("Module_B.R")

#' set data saving path
sPath1 <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part1_DownloadedData/TM_ESCA"
sPath2 <- "/home/u3808/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process"
sPath3 <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part3_AdvancedDataProcessingResult"

#' choose a cancer type
sCancer <- "ESCA"

#' =============================================================================
#' Part 2: Perform basic processing of downloaded data using Module B functions
#' =============================================================================

#' Process somatic mutation data
#' 

# path_somaticMutation <- DownloadSomaticMutationData(cancerType = sCancer,
#                               assayPlatform = "somaticMutation_DNAseq",
#                               saveFolderName = sPath1,
#                               tissueType = "TM")
# path_somaticMutation[1]
# 
# list_somaticMutation <-
#   ProcessSomaticMutationData(inputFilePath = path_somaticMutation[1],
#                              outputFileName = paste(sCancer,
#                                                     "somaticMutation",
#                                                     sep = "__"),
#                              outputFileFolder = sPath2)

# ESCA
path_somaticMutation <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part1_DownloadedData/TM_ESCA/ESCA__somaticMutation_DNAseq__TM__20220610022108__bcgsc.ca_ESCA.IlluminaHiSeq_DNASeq.1.somatic.maf.txt"
                          
list_somaticMutation <-
  ProcessSomaticMutationData(inputFilePath = path_somaticMutation,
                             outputFileName = paste(sCancer,
                                                    "somaticMutation",
                                                    sep = "__"),
                             outputFileFolder = sPath2)

#BLCA

sCancer <- "BLCA"

path_somaticMutation <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part1_DownloadedData/TM_BLCA/BLCA__somaticMutation_DNAseq__TM__20220425053859__bcgsc.ca_BLCA.IlluminaHiSeq_DNASeq.1.somatic.maf.txt"

list_somaticMutation <-
  ProcessSomaticMutationData(inputFilePath = path_somaticMutation,
                             outputFileName = paste(sCancer,
                                                    "somaticMutation",
                                                    sep = "__"),
                             outputFileFolder = sPath2)

# BRCA
sCancer <- "BRCA"

path_somaticMutation <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part1_DownloadedData/TM_BRCA/BRCA__somaticMutation_DNAseq__TM__20220425172622__gsc_BRCA_pairs.aggregated.capture.tcga.uuid.automated.somatic.maf.txt"

list_somaticMutation <-
  ProcessSomaticMutationData(inputFilePath = path_somaticMutation,
                             outputFileName = paste(sCancer,
                                                    "somaticMutation",
                                                    sep = "__"),
                             outputFileFolder = sPath2)

#CESC
sCancer <- "CESC"

path_somaticMutation <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part1_DownloadedData/TM_CESC/CESC__somaticMutation_DNAseq__TM__20220425102637__bcgsc.ca_CESC.IlluminaHiSeq_DNASeq.1.somatic.maf.txt"
list_somaticMutation <-
  ProcessSomaticMutationData(inputFilePath = path_somaticMutation,
                             outputFileName = paste(sCancer,
                                                    "somaticMutation",
                                                    sep = "__"),
                             outputFileFolder = sPath2)

# COAD
sCancer <- "COAD"

path_somaticMutation <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part1_DownloadedData/TM_COAD/COAD__somaticMutation_DNAseq__TM__20220425030149__gsc_COAD_pairs.aggregated.capture.tcga.uuid.automated.somatic.maf.tsv.txt"
list_somaticMutation <-
  ProcessSomaticMutationData(inputFilePath = path_somaticMutation,
                             outputFileName = paste(sCancer,
                                                    "somaticMutation",
                                                    sep = "__"),
                             outputFileFolder = sPath2)


# HNSC
sCancer <- "HNSC"

path_somaticMutation <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part1_DownloadedData/TM_HNSC/HNSC__somaticMutation_DNAseq__TM__20220425120851__bcgsc.ca_HNSC.IlluminaHiSeq_DNASeq.1.somatic.maf.txt"
list_somaticMutation <-
  ProcessSomaticMutationData(inputFilePath = path_somaticMutation,
                             outputFileName = paste(sCancer,
                                                    "somaticMutation",
                                                    sep = "__"),
                             outputFileFolder = sPath2)

# PAAD
sCancer <- "PAAD"

path_somaticMutation <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part1_DownloadedData/TM_PAAD/PAAD__somaticMutation_DNAseq__TM__20220425033545__bcgsc.ca_PAAD.IlluminaHiSeq_DNASeq.1.somatic.maf.txt"
list_somaticMutation <-
  ProcessSomaticMutationData(inputFilePath = path_somaticMutation,
                             outputFileName = paste(sCancer,
                                                    "somaticMutation",
                                                    sep = "__"),
                             outputFileFolder = sPath2)

# PCPG 
sCancer <- "PCPG"

path_somaticMutation <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part1_DownloadedData/TM_PCPG/PCPG__somaticMutation_DNAseq__TM__20220425102530__bcgsc.ca_PCPG.IlluminaHiSeq_DNASeq.1.somatic.maf.txt"
list_somaticMutation <-
  ProcessSomaticMutationData(inputFilePath = path_somaticMutation,
                             outputFileName = paste(sCancer,
                                                    "somaticMutation",
                                                    sep = "__"),
                             outputFileFolder = sPath2)

# PRAD 
sCancer <- "PRAD"

path_somaticMutation <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part1_DownloadedData/TM_PRAD/PRAD__somaticMutation_DNAseq__TM__20220425045607__gsc_PRAD_pairs.aggregated.capture.tcga.uuid.automated.somatic.maf.txt"
list_somaticMutation <-
  ProcessSomaticMutationData(inputFilePath = path_somaticMutation,
                             outputFileName = paste(sCancer,
                                                    "somaticMutation",
                                                    sep = "__"),
                             outputFileFolder = sPath2)

# SKCM 
sCancer <- "SKCM"

path_somaticMutation <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part1_DownloadedData/TM_SKCM/SKCM__somaticMutation_DNAseq__TM__20220426012120__SKCM_pairs.aggregated.capture.tcga.uuid.automated.somatic.maf.tsv.txt"
list_somaticMutation <-
  ProcessSomaticMutationData(inputFilePath = path_somaticMutation,
                             outputFileName = paste(sCancer,
                                                    "somaticMutation",
                                                    sep = "__"),
                             outputFileFolder = sPath2)

# THCA 
sCancer <- "THCA"

path_somaticMutation <- "~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part1_DownloadedData/TM_THCA/THCA__somaticMutation_DNAseq__TM__20220425103015__bcgsc.ca_THCA.IlluminaHiSeq_DNASeq.1.somatic.maf.txt"
list_somaticMutation <-
  ProcessSomaticMutationData(inputFilePath = path_somaticMutation,
                             outputFileName = paste(sCancer,
                                                    "somaticMutation",
                                                    sep = "__"),
                             outputFileFolder = sPath2)

#' =============================================================================
#' Part 3: Process somatic mutation data with non-silent SNVs in a gene
#' =============================================================================

#' 1. Process metastasis-related samples of 11 cancer types 
#' 

setwd("/home/u3808/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process")

BLCA <- read.delim("~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process/BLCA__somaticMutation_geneLevel.txt")
BRCA <- read.delim("~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process/BRCA__somaticMutation_geneLevel.txt")
CESC <- read.delim("~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process/CESC__somaticMutation_geneLevel.txt")
COAD <- read.delim("~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process/COAD__somaticMutation_geneLevel.txt")
HNSC <- read.delim("~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process/HNSC__somaticMutation_geneLevel.txt")
ESCA <- read.delim("~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process/ESCA__somaticMutation_geneLevel.txt")
PAAD <- read.delim("~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process/PAAD__somaticMutation_geneLevel.txt")
PCPG <- read.delim("~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process/PCPG__somaticMutation_geneLevel.txt")
PRAD <- read.delim("~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process/PRAD__somaticMutation_geneLevel.txt")
SKCM <- read.delim("~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process/SKCM__somaticMutation_geneLevel.txt")
THCA <- read.delim("~/3.TCGA/TCGA-Assembler-2/2.preprocessing/Part2_BasicDataProcessingResult/TM_mutation_process/THCA__somaticMutation_geneLevel.txt")

# Delete Silent Mutation Data
deleteSilent <- function(data2) {
  data2[data2=='Silent'] <- NA
  data2=na.omit(data2)
  nrow(data2)
  return(data2)
}

data_list <- list(BLCA,BRCA,CESC,COAD,HNSC,ESCA,PAAD,PCPG,PRAD,SKCM,THCA)
y <- lapply( data_list,deleteSilent)

# Assign values equal or more than 1 to 1
mutationprocess <- function(data) {
  matrix <- data[,3:ncol(data)]
  matrix[matrix>=1]=1
  output <- cbind(Gene=data$GeneSymbol,matrix)
  return(output)
}

z<- lapply( y,mutationprocess)


# Merge all dataframe into a matrix
my_merge <- function(df1, df2){                                # Create own merging function
  merge(df1, df2, by.x = "Gene", by.y = "Gene", all = TRUE)
}
TM_mutation <- Reduce(my_merge, z) 
TM_mutation[is.na(TM_mutation)] <- 0 

nrow(TM_mutation)
ncol(TM_mutation)

# Output the matrix
write.table (TM_mutation, file ="TM_mutation_20220620", sep ="\t", row.names =F, col.names =TRUE, quote =F)




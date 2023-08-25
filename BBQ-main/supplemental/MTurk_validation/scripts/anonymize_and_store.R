library(tidyverse)

this.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(this.dir)

SECRET_dir = "../../../SECRET/QA_bias_SECRET/qual_batch/"

anonymize = function(dat,anons){
  dat2 = merge(dat,anons)
  dat3 = dat2 %>% select(-WorkerId)
  return(dat3)
}

#--------------- CREATE ANON IDS ------------------

filenames = list.files(SECRET_dir, full.names = T)

dat = NULL
for(fil in filenames){
  temp = read.csv(fil, stringsAsFactors = F)
  if("Answer.response_ex3" %in% colnames(temp)){
    temp = temp %>% select(-Answer.response_ex3)
  }
  dat = rbind(dat,temp)
}

# create_ids
len_anons = 0
WorkerId = unique(dat$WorkerId)
anon_id = c((1+len_anons):(length(WorkerId)+len_anons))
updated_anons = cbind(WorkerId,anon_id)

write.csv(updated_anons,"../../../SECRET/QA_bias_SECRET/anon_ids.csv", row.names=FALSE)

#--------------- ANONYMIZE DATA AND STORE ------------------

anon_ids = read.csv("../../../SECRET/QA_bias_SECRET/anon_ids.csv")

anon_dat = anonymize(dat,anon_ids)

write.csv(anon_dat,"../data/qual_batch.csv", row.names=FALSE)

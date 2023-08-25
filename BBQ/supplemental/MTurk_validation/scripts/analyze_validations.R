library(tidyverse)

this.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(this.dir)

#dat <- read.csv("../../../SECRET/QA_bias_SECRET/validations_round1/Batch_4466786_batch_results.csv", stringsAsFactors = F) # round 1.1
#dat <- read.csv("../../../SECRET/QA_bias_SECRET/validations_round1/Batch_4467422_batch_results.csv", stringsAsFactors = F) # round 1.2
#dat <- read.csv("../../../SECRET/QA_bias_SECRET/validations_round1/Batch_4467511_batch_results.csv", stringsAsFactors = F) # round 1.3
#dat <- read.csv("../../../SECRET/QA_bias_SECRET/validations_round1/Batch_4468041_batch_results.csv", stringsAsFactors = F) # round 1.4
#dat <- read.csv("../../../SECRET/QA_bias_SECRET/validations_round2/Batch_4485563_batch_results.csv", stringsAsFactors = F) # round 2.1
#dat <- read.csv("../../../SECRET/QA_bias_secret/validations_round2/Batch_4485631_batch_results.csv", stringsAsFactors = F)
#dat <- read.csv("../../../SECRET/QA_bias_secret/validations_round2/Batch_4485766_batch_results.csv", stringsAsFactors = F)
#dat <- read.csv("../../../SECRET/QA_bias_secret/validations_round3/Batch_4508583_batch_results.csv", stringsAsFactors = F)

filenames = list.files("../../../SECRET/QA_bias_SECRET/validations_finalval", pattern="*.csv", full.names = T)
dat = NULL
for(f in filenames){
  temp = read.csv(f, stringsAsFactors = F)
  temp <- temp %>% select(-contains("checkbox"))
  dat=rbind(dat,temp)
}

cond_info <- read.csv("all_examples_uid_finalval.csv", stringsAsFactors = F)
cond_info <- cond_info %>% select(u_id,question_index,category,context_condition,question_polarity) %>% rename("uid" = u_id)

dat2 <- dat %>%
  select(contains('Answer'), contains('Input'), WorkerId, WorkTimeInSeconds) %>%
  select(-Answer.useragent, -Answer.response_ex1, -Answer.response_ex2)
colnames(dat2)<-gsub("Input.","",colnames(dat2))

mean(dat2$WorkTimeInSeconds/60)
comments = unique(dat2$Answer.comments)
print(comments)

table(dat2$WorkerId)

# anonymize and store
anons <- read.csv("../../../SECRET/QA_bias_SECRET/anon_ids.csv", stringsAsFactors = F)
anon_dat <- merge(dat2,anons,by="WorkerId") %>%
  select(-WorkerId)
write.csv(anon_dat, "../data/validations_finalval.csv",row.names = F)

dat_formatted = NULL
for(i in c(1:5)){
  q = paste0("q",i)
  dat_temp <- dat2 %>%
    select(contains(q),WorkerId)
  colnames(dat_temp)<-gsub(q,"",colnames(dat_temp))
  colnames(dat_temp)<-gsub("_","",colnames(dat_temp))
  dat_temp$q_num = q
  if(!"Answer.checkbox" %in% colnames(dat_temp)){dat_temp$Answer.checkbox = NA}
  dat_formatted = rbind(dat_formatted,dat_temp)
}

dat_form_2 <- dat_formatted %>%
  mutate(response = case_when(Answer. == "ans_2" ~ 2,
                              Answer. == "ans_1" ~ 1,
                              Answer. == "ans_0" ~ 0)) %>%
  mutate(correct = case_when(response==label ~ 1,
                             response!=label ~0))

# overall validation rate
mean(dat_form_2$correct, na.rm=T)

# check for issues reported
#View(dat_form_2 %>% filter(Answer.checkbox!=""))

# check by worker for outliers in response tendency
dat_form_2 %>%
  group_by(WorkerId)%>%
  summarise(pct_corr = mean(correct,na.rm=T),
            count=n())

# accuracy on just fillers -- actual accuracy measure
dat_form_2 %>%
  filter(is.na(exid)) %>%
  group_by(WorkerId)%>%
  summarise(pct_corr = mean(correct,na.rm=T),
            count=n())

# see accuracy just for the non fillers
dat_test_acc <- dat_form_2 %>%
  filter(!is.na(exid))
mean(dat_test_acc$correct)

# per item accuracy
dat_item_acc <- dat_test_acc %>%
  group_by(uid)%>%
  summarise(acc=mean(correct),
            count=n())

# check by each question
dat_form_3 <- dat_form_2 %>%
  group_by(input,question,uid,ans0,ans1,ans2,label,cat)%>%
  summarise(pct_corr = mean(correct),
            resps = paste0(as.character(response), collapse=","),
            count=n())

# get IAA
dat_kappa <- dat_test_acc %>%
  select(WorkerId,uid,response)%>%
  spread(WorkerId,response)%>%
  select(-uid)
kripp.alpha(data.matrix(t(dat_kappa)))

# check how many templates already pass validation
nrow(filter(dat_form_3,pct_corr>=0.8))/nrow(dat_form_3)

dat_form <- merge(dat_form_3, cond_info, by = "uid")
View(dat_form)

write.csv(dat_form,"../results_analysis/results_finalval.csv",row.names = F)


# ---------------------------------------
# check all validation rates
fils = list.files("../results_analysis/round2",pattern="*.csv",full.names=T)
dat_all = NULL
for(f in fils){
  temp = read.csv(f, stringsAsFactors = F)
  dat_all = rbind(dat_all, temp)
}

dat_all2<-dat_all%>%
  filter(context_condition %in% c("disambig", "ambig")) %>%
  mutate(pass = case_when(decision=="pass" ~ 1,
                          decision!="pass"~0))%>%
  filter(pass==1)%>%
  mutate(acc = case_when(!is.na(new_pct_corr) ~ new_pct_corr,
                         is.na(new_pct_corr) ~ pct_corr))

# ----- check fillers --------
dat.fill <- dat_all %>%
  filter(!context_condition %in% c("disambig", "ambig"))
  

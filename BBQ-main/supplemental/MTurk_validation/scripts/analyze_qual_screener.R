library(tidyverse)

this.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(this.dir)

# read in data
dat = read.csv("../data/qual_batch.csv", stringsAsFactors = F)

dat2 <- dat %>%
  # only take responses where everything was answered and workers agreed to do future HITs
  filter(Answer.numanswered==6,
         Answer.future_HITs=='yes')%>%
  # make the dataframe smaller and more manageable
  select(anon_id,contains("Answer.q"))%>%
  # turn into longform
  gather("Q_num", "response", -anon_id)%>%
  # easier/shorter values to work with
  mutate(Q_num = str_remove(Q_num, "Answer."))%>%
  # calculate if the answer is correct, based on separate answer key
  mutate(is_corr = ifelse(Q_num %in% c('q1','q3','q5') & response=='ans_3', 1,
                          ifelse(Q_num %in% c('q2','q4') & response == 'ans_1', 1, 0)))%>%
  # get accuracy by each worker
  group_by(anon_id)%>%
  summarise(accuracy = mean(is_corr)) %>%
  # only take workers who got all qual questions right
  filter(accuracy == 1)

length(unique(dat2$anon_id))

write.csv(dat2, "../included_workers.csv", row.names = F)

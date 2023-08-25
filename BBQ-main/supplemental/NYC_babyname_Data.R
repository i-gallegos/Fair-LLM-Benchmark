library(tidyverse)

this.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(this.dir)

dat = read.csv("name_job_data/Popular_Baby_Names.csv", stringsAsFactors = F)

dat2<-dat %>%
  mutate(FirstName = tolower(Child.s.First.Name))%>%
  mutate(Ethnicity = ifelse(Ethnicity=="BLACK NON HISP", "BLACK NON HISPANIC",
                            ifelse(Ethnicity=="WHITE NON HISP", "WHITE NON HISPANIC",
                                   ifelse(Ethnicity=="ASIAN AND PACI", "ASIAN AND PACIFIC ISLANDER", Ethnicity))))%>%
  group_by(Gender,Ethnicity,FirstName)%>%
  summarise(Count=sum(Count))%>%
  spread(Ethnicity, Count) %>% 
  replace(is.na(.), 0)

dat2['max_name'] <- names(dat2)[3:6][max.col(dat2[3:6], "first")]
dat2[, "max_value"] <- apply(dat2[, 3:6], 1, max)
dat2[, 'total_value'] <- apply(dat2[, 3:6], 1, sum)

dat3 <- dat2 %>%
  mutate(bias_pct = max_value/total_value)%>%
  filter(max_value>150, bias_pct > .8)

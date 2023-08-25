# --------------------------------------
# This script creates metadata useful for analyzing BBQ results
# It outputs a csv file that contains information that can be matched
# to a results file using the example id and the category information
# --------------------------------------

library(tidyverse)
library(stringr)
library(jsonlite)

this.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(this.dir)

read_json <- function(file){
  con <- file(file, open = "r")
  on.exit(close(con))
  jsonlite::stream_in(con, verbose = FALSE)
}

unknowns = c("unknown", "cannot be determined", "can't be determined", "not answerable",
             "not known", "not enough info", "not enough information", "cannot answer",
             "can't answer", "undetermined")

filenames = list.files("../data/",full.names = T, pattern = "*.jsonl")

dat=NULL
for(f in filenames){
  temp = read_json(f)
  ans0_info = temp$answer_info["ans0"] %>% mutate(ans0 = purrr::map(ans0, setNames, c("ans0_text", "ans0_info"))) %>% unnest_wider(ans0)
  ans1_info = temp$answer_info["ans1"] %>% mutate(ans1 = purrr::map(ans1, setNames, c("ans1_text", "ans1_info"))) %>% unnest_wider(ans1)
  ans2_info = temp$answer_info["ans2"] %>% mutate(ans2 = purrr::map(ans2, setNames, c("ans2_text", "ans2_info"))) %>% unnest_wider(ans2)
  stereotyped_groups = temp$additional_metadata["stereotyped_groups"] %>% 
    mutate(stereotyped_groups = as.character(stereotyped_groups))
  temp = temp %>% select(-answer_info, -additional_metadata)
  temp = cbind(temp,cbind(ans0_info,cbind(ans1_info,cbind(ans2_info,stereotyped_groups))))
  dat=rbind(dat,temp)
}


# first add missing info
check_missing <- dat %>%
  filter(stereotyped_groups=="list()")
unique(check_missing$category)

templates_files = list.files("../templates/", pattern="*.csv", full.names = T)
st_group_data = NULL
for(i in c(1:length(templates_files))){
  if(!grepl("vocab",templates_files[i]) & !grepl("_x_",templates_files[i]) & !grepl("Filler",templates_files[i])){
    temp = read.csv(templates_files[i],stringsAsFactors = F)
    temp2 <- temp %>%
      select(Category,Known_stereotyped_groups,Q_id,Relevant_social_values)%>%
      rename("category"=Category,
             #"stereotyped_groups" = Known_stereotyped_groups,
             "question_index" = Q_id)%>%
      mutate(question_index = as.character(question_index))
    st_group_data = rbind(st_group_data,temp2)
  }
}

st_group_data2 = st_group_data%>%
  mutate(category = ifelse(category=="GenderIdentity","Gender_identity",
                           ifelse(category=="PhysicalAppearance", "Physical_appearance",
                                  ifelse(category=="RaceEthnicity", "Race_ethnicity",
                                         ifelse(category=="Religion ", "Religion",
                                                ifelse(category=="SexualOrientation", "Sexual_orientation", 
                                                       ifelse(category=="DisabilityStatus", "Disability_status", category)))))))%>%
  group_by(category,question_index,Known_stereotyped_groups,Relevant_social_values)%>%
  summarise(count=n())%>%
  select(-count)

dat4 <- merge(dat,st_group_data2,by=c("category","question_index"),all=T)

# for non intersectional templates
dat_base<-dat4 %>% 
  filter(!grepl("_x_",category))%>%
  select(-stereotyped_groups)%>% # this is redundant now, getting rid of it to avoid confusion
  mutate(Known_stereotyped_groups = gsub('"|\\[|\\]', '', Known_stereotyped_groups))%>%
  mutate(ans0_text = gsub('\\}|\\{', '', ans0_text))%>%
  mutate(ans1_text = gsub('\\}|\\{', '', ans1_text))%>%
  mutate(ans2_text = gsub('\\}|\\{', '', ans2_text))%>%
  mutate(ProperName = case_when(as.integer(question_index)>25 ~ "name",
                                as.integer(question_index)<=25 ~ "label"))%>%
  mutate(ans0_info = ifelse(ans0_info=="man"|ans0_info=="boy", "M",
                            ifelse(ans0_info=="woman"|ans0_info=="girl","F",ans0_info)))%>%
  mutate(ans1_info = ifelse(ans1_info=="man"|ans1_info=="boy", "M",
                            ifelse(ans1_info=="woman"|ans1_info=="girl","F",ans1_info)))%>%
  mutate(ans2_info = ifelse(ans2_info=="man"|ans2_info=="boy", "M",
                            ifelse(ans2_info=="woman"|ans2_info=="girl","F",ans2_info)))%>%
  mutate(Known_stereotyped_groups = ifelse(tolower(Known_stereotyped_groups)=="man"|tolower(Known_stereotyped_groups)=="boy"|tolower(Known_stereotyped_groups)=="men", "M",
                                           ifelse(tolower(Known_stereotyped_groups)=="woman"|tolower(Known_stereotyped_groups)=="women"|tolower(Known_stereotyped_groups)=="girl"|tolower(Known_stereotyped_groups)=="girls","F",Known_stereotyped_groups)))%>%
  mutate(Known_stereotyped_groups = ifelse(tolower(Known_stereotyped_groups)=="transgender women, transgender men"|tolower(Known_stereotyped_groups)=="transgender men"|tolower(Known_stereotyped_groups)=="transgender women"|tolower(Known_stereotyped_groups)=="transgender women, transgender men, trans","trans",Known_stereotyped_groups))%>%
  mutate(ans0_info = str_remove(ans0_info, "M-"),
         ans0_info = str_remove(ans0_info, "F-"))%>%
  mutate(ans1_info = str_remove(ans1_info, "M-"),
         ans1_info = str_remove(ans1_info, "F-"))%>%
  mutate(ans2_info = str_remove(ans2_info, "M-"),
         ans2_info = str_remove(ans2_info, "F-"))%>%
  mutate(Known_stereotyped_groups = ifelse(category=="Disability status", "disabled", Known_stereotyped_groups))%>%
  mutate(Known_stereotyped_groups = ifelse(Known_stereotyped_groups=="low SES", "lowSES",
                                           ifelse(Known_stereotyped_groups=="high SES", "highSES", Known_stereotyped_groups)))%>%
  mutate(target_loc_0 = case_when(str_detect(tolower(Known_stereotyped_groups), tolower(ans0_info)) ~ 1,
                                  !(str_detect(tolower(Known_stereotyped_groups), tolower(ans0_info))) ~ 0))%>%
  mutate(target_loc_1 = case_when(str_detect(tolower(Known_stereotyped_groups), tolower(ans1_info)) ~ 1,
                                  !(str_detect(tolower(Known_stereotyped_groups), tolower(ans1_info))) ~ 0))%>%
  mutate(target_loc_2 = case_when(str_detect(tolower(Known_stereotyped_groups), tolower(ans2_info)) ~ 1,
                                  !(str_detect(tolower(Known_stereotyped_groups), tolower(ans2_info))) ~ 0))%>%
  # fix issue in Age category (word 'old' in 'nonold')
  mutate(target_loc_0 = ifelse(category=="Age",ifelse((Known_stereotyped_groups=="nonOld" & ans0_info=="nonOld") | (Known_stereotyped_groups=='old' & ans0_info=='old'), 1, 0),target_loc_0),
         target_loc_1 = ifelse(category=="Age",ifelse((Known_stereotyped_groups=="nonOld" & ans1_info=="nonOld") | (Known_stereotyped_groups=='old' & ans1_info=='old'),1,0),target_loc_1),
         target_loc_2 = ifelse(category=="Age",ifelse((Known_stereotyped_groups=="nonOld" & ans2_info=="nonOld") | (Known_stereotyped_groups=='old' & ans2_info=='old'),1,0),target_loc_2))%>%
  # fix issue in nationality category (need to check text)
  mutate(target_loc_0 = ifelse(category=="Nationality" & str_detect(tolower(Known_stereotyped_groups), tolower(ans0_text)),1,target_loc_0),
         target_loc_1 = ifelse(category=="Nationality" & str_detect(tolower(Known_stereotyped_groups), tolower(ans1_text)),1,target_loc_1),
         target_loc_2 = ifelse(category=="Nationality" & str_detect(tolower(Known_stereotyped_groups), tolower(ans2_text)),1,target_loc_2))%>%
  # fix issue in race category (separate 'african' / 'african american')
  mutate(target_loc_0 = ifelse(ans0_text=="African" & str_detect(tolower(Known_stereotyped_groups), tolower("african american")),0,target_loc_0),
         target_loc_1 = ifelse(ans1_text=="African" & str_detect(tolower(Known_stereotyped_groups), tolower("african american")),0,target_loc_1),
         target_loc_2 = ifelse(ans2_text=="African" & str_detect(tolower(Known_stereotyped_groups), tolower("african american")),0,target_loc_2))%>%
  # fix issue in identifying trans target
  mutate(target_loc_0 = ifelse(Known_stereotyped_groups=="trans" & grepl("^trans",ans0_info),1,target_loc_0),
         target_loc_1 = ifelse(Known_stereotyped_groups=="trans" & grepl("^trans",ans1_info),1,target_loc_1),
         target_loc_2 = ifelse(Known_stereotyped_groups=="trans" & grepl("^trans",ans2_info),1,target_loc_2))%>%
  mutate(target_loc = case_when(target_loc_0 == 1 ~ 0,
                                target_loc_1 == 1 ~ 1,
                                target_loc_2 == 1 ~ 2))%>%
  rename("label_type" = ProperName)

# in the non-neg examples, the target is actually the non-target, here looking for the other answer option that's not target_loc and not "unknown"
dat_base_target_loc_corrected <- dat_base %>%
  mutate(new_target_loc = case_when(question_polarity=="nonneg"&target_loc==0&ans1_info!="unknown"~1,
                                    question_polarity=="nonneg"&target_loc==0&ans2_info!="unknown"~2,
                                    question_polarity=="nonneg"&target_loc==1&ans0_info!="unknown"~0,
                                    question_polarity=="nonneg"&target_loc==1&ans2_info!="unknown"~2,
                                    question_polarity=="nonneg"&target_loc==2&ans0_info!="unknown"~0,
                                    question_polarity=="nonneg"&target_loc==2&ans1_info!="unknown"~1,
                                    question_polarity=="neg" ~ target_loc))%>%
  # make sure target loc wasn't identified for more than one answer option
  mutate(new_target_loc = ifelse(target_loc_0+target_loc_1+target_loc_2 > 1, NA,new_target_loc))

# should come back as 0, is actually 16 and these questions will be removed in later analysis
length(dat_base_target_loc_corrected%>%filter(is.na(new_target_loc))%>%pull(question_index))

table(dat_base$target_loc)
table(dat_base_target_loc_corrected$new_target_loc)

dat_base_selected <- dat_base_target_loc_corrected %>%
  select(category,question_index,example_id,new_target_loc,label_type,Known_stereotyped_groups,Relevant_social_values)%>%
  rename(target_loc=new_target_loc)


# ------------------------------------------------------------
# for intersectional templates
templates_files = list.files("../templates/", pattern="*.csv", full.names = T)
st_group_data = NULL
for(i in c(1:length(templates_files))){
  if(grepl("_x_",templates_files[i])){
    temp = read.csv(templates_files[i],stringsAsFactors = F)
    temp2 <- temp %>%
      select(Category,Known_stereotyped_race,Known_stereotyped_var2,Q_id,Relevant_social_values,Proper_nouns_only)%>%
      rename("category"=Category,
             "question_index" = Q_id)%>%
      mutate(question_index = as.character(question_index))
    st_group_data = rbind(st_group_data,temp2)
  }
}

st_group_data2 = st_group_data%>%
  mutate(category=ifelse(category=='Gender_x_race','Race_x_gender',category))%>%
  group_by(category,question_index,Known_stereotyped_race,Known_stereotyped_var2,Relevant_social_values,Proper_nouns_only)%>%
  summarise(count=n())%>%
  select(-count)

dat4 <- merge(dat,st_group_data2,by=c("category","question_index"),all=T) %>%
  filter(!is.na(example_id))

dat.race_x_gender<-dat4 %>% 
  filter(category=="Race_x_gender")%>%
  select(-stereotyped_groups)%>% # this is redundant now, getting rid of it to avoid confusion
  mutate(label_type = case_when(Proper_nouns_only==T ~ "name",
                                is.na(Proper_nouns_only) ~ "label"))%>%
  select(-Proper_nouns_only)%>%
  mutate(Known_stereotyped_race = gsub('"|\\[|\\]', '', Known_stereotyped_race))%>%
  mutate(ans0_gender = ifelse(ans0_info=='unknown','unknown',substr(ans0_info,1,1)),
         ans0_race = ifelse(ans0_info=='unknown','unknown',substr(ans0_info,3,length(ans0_info))),
         ans1_gender = ifelse(ans1_info=='unknown','unknown',substr(ans1_info,1,1)),
         ans1_race = ifelse(ans1_info=='unknown','unknown',substr(ans1_info,3,length(ans1_info))),
         ans2_gender = ifelse(ans2_info=='unknown','unknown',substr(ans2_info,1,1)),
         ans2_race = ifelse(ans2_info=='unknown','unknown',substr(ans2_info,3,length(ans2_info))))%>%
  mutate(ans0_gender = ifelse(str_detect(ans0_text," man"),"M",
                              ifelse(str_detect(ans0_text," woman"), "F",ans0_gender)),
         ans1_gender = ifelse(str_detect(ans1_text," man"),"M",
                              ifelse(str_detect(ans1_text," woman"), "F",ans1_gender)),
         ans2_gender = ifelse(str_detect(ans2_text," man"),"M",
                              ifelse(str_detect(ans2_text," woman"), "F",ans2_gender)))%>%
  mutate(ans0_gender = ifelse(ans0_gender==""&str_detect(ans0," man"),"M",
                              ifelse(ans0_gender==""&str_detect(ans0," woman"),"F",ans0_gender)),
         ans1_gender = ifelse(ans1_gender==""&str_detect(ans1," man"),"M",
                              ifelse(ans1_gender==""&str_detect(ans1," woman"),"F",ans1_gender)),
         ans2_gender = ifelse(ans2_gender==""&str_detect(ans2," man"),"M",
                              ifelse(ans2_gender==""&str_detect(ans2," woman"),"F",ans2_gender)),
         ans0_race = ifelse(ans0_race=="","Latino",ans0_race),
         ans1_race = ifelse(ans1_race=="","Latino",ans1_race),
         ans2_race = ifelse(ans2_race=="","Latino",ans2_race))%>%
  mutate(corr_ans_race = case_when(label==0 ~ ans0_race,
                                   label==1 ~ ans1_race,
                                   label==2 ~ ans2_race),
         corr_ans_gender = case_when(label==0 ~ ans0_gender,
                                     label==1 ~ ans1_gender,
                                     label==2 ~ ans2_gender))%>%
  mutate(corr_ans_aligns_race = ifelse(str_detect(Known_stereotyped_race,corr_ans_race),1,0),
         corr_ans_aligns_gender = ifelse(str_detect(Known_stereotyped_var2,corr_ans_gender),1,0))%>%
  mutate(race_condition = ifelse(ans0_race==ans1_race|ans1_race==ans2_race|ans0_race==ans2_race,"Match Race","Mismatch Race"),
         gender_condition = ifelse(ans0_gender==ans1_gender|ans1_gender==ans2_gender|ans0_gender==ans2_gender,"Match Gender","Mismatch Gender"))%>%
  mutate(full_cond = paste(race_condition,gender_condition,sep="\n ")) %>%
  mutate(target_loc = ifelse(substr(Known_stereotyped_var2,1,1)==ans0_gender & str_detect(Known_stereotyped_race,substr(ans0_race,3,length(ans0_race))),0,
                             ifelse(substr(Known_stereotyped_var2,1,1)==ans1_gender & str_detect(Known_stereotyped_race,substr(ans1_race,3,length(ans1_race))),1,
                                    ifelse(substr(Known_stereotyped_var2,1,1)==ans2_gender & str_detect(Known_stereotyped_race,substr(ans2_race,3,length(ans2_race))),2,NA))))%>%
  rename("corr_ans_aligns_var2" = corr_ans_aligns_gender)

# in the non-neg examples, the target is actually the non-target, here looking for the other answer option that's not target_loc and not "unknown"
dat.race_x_gender_target_loc_corrected <- dat.race_x_gender %>%
  mutate(new_target_loc = case_when(question_polarity=="nonneg"&target_loc==0&ans1_gender!="unknown"~1,
                                    question_polarity=="nonneg"&target_loc==0&ans2_gender!="unknown"~2,
                                    question_polarity=="nonneg"&target_loc==1&ans0_gender!="unknown"~0,
                                    question_polarity=="nonneg"&target_loc==1&ans2_gender!="unknown"~2,
                                    question_polarity=="nonneg"&target_loc==2&ans0_gender!="unknown"~0,
                                    question_polarity=="nonneg"&target_loc==2&ans1_gender!="unknown"~1,
                                    question_polarity=="neg" ~ target_loc))

# should come back as 0
length(dat.race_x_gender_target_loc_corrected%>%filter(is.na(new_target_loc))%>%pull(question_index))

dat_racegen_selected <- dat.race_x_gender_target_loc_corrected %>%
  select(category,question_index,example_id,new_target_loc,label_type,
         Known_stereotyped_race,Known_stereotyped_var2,Relevant_social_values,
         corr_ans_aligns_var2,corr_ans_aligns_race,full_cond)%>%
  rename(target_loc=new_target_loc)

table(dat.race_x_gender$full_cond)
table(dat_racegen_selected$full_cond) # should not change from above
table(dat.race_x_gender$target_loc)
table(dat_racegen_selected$target_loc) # should change slightly from above

dat.race_x_ses<-dat4 %>% 
  filter(category=="Race_x_SES")%>%
  select(-stereotyped_groups)%>% # this is redundant now, getting rid of it to avoid confusion
  mutate(label_type = case_when(Proper_nouns_only==T ~ "name",
                                is.na(Proper_nouns_only) ~ "label"))%>%
  select(-Proper_nouns_only)%>%
  mutate(Known_stereotyped_race = gsub('"|\\[|\\]', '', Known_stereotyped_race))%>%
  mutate(ans0_gender = ifelse(ans0_info=='unknown','unknown',substr(ans0_info,1,6)),
         ans0_race = ifelse(ans0_info=='unknown','unknown',substr(ans0_info,8,length(ans0_info))),
         ans1_gender = ifelse(ans1_info=='unknown','unknown',substr(ans1_info,1,6)),
         ans1_race = ifelse(ans1_info=='unknown','unknown',substr(ans1_info,8,length(ans1_info))),
         ans2_gender = ifelse(ans2_info=='unknown','unknown',substr(ans2_info,1,6)),
         ans2_race = ifelse(ans2_info=='unknown','unknown',substr(ans2_info,8,length(ans2_info))))%>%
  mutate(ans0_race = gsub("M-","",ans0_race),
         ans1_race = gsub("M-","",ans1_race),
         ans2_race = gsub("M-","",ans2_race),
         ans0_race = gsub("F-","",ans0_race),
         ans1_race = gsub("F-","",ans1_race),
         ans2_race = gsub("F-","",ans2_race))%>%
  mutate(ans0_race = gsub("-","",ans0_race),
         ans1_race = gsub("-","",ans1_race),
         ans2_race = gsub("-","",ans2_race),
         ans0_gender = ifelse(ans0_gender=="highSE","highSES",ans0_gender),
         ans1_gender = ifelse(ans1_gender=="highSE","highSES",ans1_gender),
         ans2_gender = ifelse(ans2_gender=="highSE","highSES",ans2_gender))%>%
  mutate(corr_ans_race = case_when(label==0 ~ ans0_race,
                                   label==1 ~ ans1_race,
                                   label==2 ~ ans2_race),
         corr_ans_gender = case_when(label==0 ~ ans0_gender,
                                     label==1 ~ ans1_gender,
                                     label==2 ~ ans2_gender))%>%
  mutate(corr_ans_aligns_race = ifelse(str_detect(Known_stereotyped_race,corr_ans_race),1,0),
         corr_ans_aligns_gender = ifelse(str_detect(Known_stereotyped_var2,corr_ans_gender),1,0))%>%
  mutate(race_condition = ifelse(ans0_race==ans1_race|ans1_race==ans2_race|ans0_race==ans2_race,"Match Race","Mismatch Race"),
         SES_condition = ifelse(ans0_gender==ans1_gender|ans1_gender==ans2_gender|ans0_gender==ans2_gender,"Match SES","Mismatch SES"))%>%
  mutate(Known_stereotyped_race = ifelse(Known_stereotyped_race=="Black, African American, Latino, Hispanic" & (ans0_race=="Hispanic"|ans0_race=="Latino"|ans1_race=="Hispanic"|ans1_race=="Latino"|ans2_race=="Hispanic"|ans2_race=="Latino"),"Hispanic, Latino", 
                                         ifelse(Known_stereotyped_race=="Black, African American, Latino, Hispanic" & (ans0_race=="Black"|ans0_race=="African American"|ans1_race=="Black"|ans1_race=="African American"|ans2_race=="Black"|ans2_race=="African American"), "Black, African American", Known_stereotyped_race)))%>%
  mutate(full_cond = paste(race_condition,SES_condition,sep="\n ")) %>%
  mutate(target_loc = ifelse(substr(Known_stereotyped_var2,1,3)==substr(ans0_gender,1,3) & str_detect(Known_stereotyped_race,ans0_race),0,
                             ifelse(substr(Known_stereotyped_var2,1,3)==substr(ans1_gender,1,3) & str_detect(Known_stereotyped_race,ans1_race),1,
                                    ifelse(substr(Known_stereotyped_var2,1,3)==substr(ans2_gender,1,3) & str_detect(Known_stereotyped_race,ans2_race),2,NA))))%>%
  rename("corr_ans_aligns_var2" = corr_ans_aligns_gender)

# in the non-neg examples, the target is actually the non-target, here looking for the other answer option that's not target_loc and not "unknown"
dat.race_x_ses_target_loc_corrected <- dat.race_x_ses %>%
  mutate(new_target_loc = case_when(question_polarity=="nonneg"&target_loc==0&ans1_gender!="unknown"~1,
                                    question_polarity=="nonneg"&target_loc==0&ans2_gender!="unknown"~2,
                                    question_polarity=="nonneg"&target_loc==1&ans0_gender!="unknown"~0,
                                    question_polarity=="nonneg"&target_loc==1&ans2_gender!="unknown"~2,
                                    question_polarity=="nonneg"&target_loc==2&ans0_gender!="unknown"~0,
                                    question_polarity=="nonneg"&target_loc==2&ans1_gender!="unknown"~1,
                                    question_polarity=="neg" ~ target_loc))

# should come back as 0
length(dat.race_x_gender_target_loc_corrected%>%filter(is.na(new_target_loc))%>%pull(question_index))

dat_raceses_selected <- dat.race_x_ses_target_loc_corrected %>%
  select(category,question_index,example_id,new_target_loc,label_type,
         Known_stereotyped_race,Known_stereotyped_var2,Relevant_social_values,
         corr_ans_aligns_var2,corr_ans_aligns_race,full_cond)%>%
  rename(target_loc=new_target_loc)


table(dat.race_x_ses$full_cond)
table(dat_raceses_selected$full_cond) # should be the same as above
table(dat.race_x_ses$target_loc)
table(dat_raceses_selected$target_loc) # should be slightly different from above

dat_ints <- rbind(dat_racegen_selected,dat_raceses_selected)
dat_ints$Known_stereotyped_groups = NA

dat_base_selected$full_cond = NA
dat_base_selected$Known_stereotyped_race = NA
dat_base_selected$Known_stereotyped_var2 = NA
dat_base_selected$corr_ans_aligns_var2 = NA
dat_base_selected$corr_ans_aligns_race = NA

all_metadata = rbind(dat_ints,dat_base_selected)

write.csv(all_metadata,"additional_metadata.csv",row.names = F)

#aside from the 'they', 'their'....
#We also want to use gender-neutral terms
import re

#EVA: This should be done different, basically make sure I replace the word and not if it part of a word!
def genderneutral(sentences):
    new_sentences=[]
    for s in sentences:
        #################################
        # 1. CHANGE INTO GENDER NEUTRAL #
        #################################
        #chairman/woman
        s = s.replace('chairman', 'chairperson')
        s = s.replace('chairmen', 'chairpeople')
        s = s.replace('chairwoman', 'chairperson')
        s = s.replace('chairwomen', 'chairpeople')
        #anchorman/woman
        s = s.replace('anchorman', 'anchor')
        s = s.replace('anchormen', 'anchors')
        s = s.replace('anchorwoman', 'anchor')
        s = s.replace('anchorwomen', 'anchors')
        #congresswoman/congressman
        s = s.replace('congressman', 'member of congress')
        s = s.replace('congressmen', 'members of congress')
        s = s.replace('congresswoman', 'member of congress')
        s = s.replace('congresswomen', 'members of congress')
        #policeman/woman
        s = s.replace('policeman', 'police officer')
        s = s.replace('policemen', 'police officers')
        s = s.replace('policewoman', 'police officer')
        s = s.replace('policewomen', 'police officers')
        #spokesman/woman
        s = s.replace('spokesman', 'spokesperson')
        s = s.replace('spokesmen', 'spokespersons')
        s = s.replace('spokeswoman', 'spokesperson')
        s = s.replace('spokeswomen', 'spokespersons')
        #steward/stewardess
        s = s.replace('steward', 'flight attendant')
        s = s.replace('stewards', 'flight attendants')
        s = s.replace('stewardess', 'flight attendant')
        s = s.replace('stewardesses', 'flight attendants')
        #headmaster/mistress
        s = s.replace('headmaster', 'principal')
        s = s.replace('headmasters', 'principals')
        s = s.replace('headmistress', 'principal')
        s = s.replace('headmistresses', 'principals')
        #business man/woman
        s = s.replace('businessman', 'business person')
        s = s.replace('businessmen', 'business people')
        s = s.replace('businesswoman', 'business person')
        s = s.replace('businesswomen', 'business persons')
        #postman/postwoman
        s = s.replace('postman', 'mail carrier')
        s = s.replace('postmen', 'mail carriers')
        s = s.replace('postwoman', 'mail carrier')
        s = s.replace('postwomen', 'mail carriers')
        #mailman/mailwoman
        s = s.replace('mailman', 'mail carrier')
        s = s.replace('mailmen', 'mail carriers')
        s = s.replace('mailwoman', 'mail carrier')
        s = s.replace('mailwomen', 'mail carriers')
        #salesman/saleswoman
        s = s.replace('salesman', 'salesperson')
        s = s.replace('salesmen', 'salespersons')
        s = s.replace('saleswoman', 'salesperson')
        s = s.replace('saleswomen', 'salespersons')
        #fireman/firewoman
        s = s.replace('fireman', 'firefighter')
        s = s.replace('firemen', 'firefighters')
        s = s.replace('firewoman', 'firefighter')
        s = s.replace('firewomen', 'firefighter')
        #barman/barwoman
        s = s.replace('barman', 'bartender')
        s = s.replace('barmen', 'bartenders')
        s = s.replace('barwoman', 'bartender')
        s = s.replace('barwomen', 'bartenders')
        #cleaning lady
        s = s.replace('cleaning man', 'cleaner')
        s = s.replace('cleaning lady', 'cleaners')
        s = s.replace('cleaning men', 'cleaner')
        s = s.replace('cleaning ladies', 'cleaners')
        #foreman/woman
        s = s.replace('foreman', 'supervisor')
        s = s.replace('foremen', 'supervisors')
        s = s.replace('forewoman', 'supervisor')
        s = s.replace('forewomen', 'supervisors')

        #######################################
        # 2. AVOID UNNECESSARY FEMININE FORMS #
        #######################################
        #actor/actress
        s = s.replace('actress', 'actor')
        s = s.replace('actresses', 'actors')
        #hero/heroine
        s = s.replace('heroine', 'hero')
        s = s.replace('heroines', 'heroes')
        #comedian/comedienne
        s = s.replace('comedienne', 'comedian')
        s = s.replace('comediennes', 'comedians')
        #executrix/executor
        s = s.replace('executrix', 'executor')
        s = s.replace('executrices', 'executors')
        s = s.replace('executrixes', 'executors')
        #poetess/poet
        s = s.replace('poetess', 'poet')
        s = s.replace('poetesses', 'poets')
        #usherette/usher
        s = s.replace('usherette', 'usher')
        s = s.replace('usherettes', 'ushers')
        #authoress/author
        s = s.replace('authoress', 'author')
        s = s.replace('authoresses', 'authors')
        #boss lady
        s = s.replace('boss lady', 'boss')
        s = s.replace('boss ladies', 'bosses')
        #boss lady
        s = s.replace('waitress', 'waiter')
        s = s.replace('waitresses', 'waiters')
        #################################
        # 3. AVOIDANCE OF GENERIC 'MAN' #
        #################################
        #average man
        s = s.replace('average man', 'average person')
        s = s.replace('average men', 'average people')
        #best man for the job
        s = s.replace('best man for the job', ' best person for the job')
        s = s.replace('best men for the job', ' best people for the job')
        #layman
        s = s.replace('layman', 'layperson')
        s = s.replace('laymen', 'laypeople')
        #man and wife
        s = s.replace(' man and wife', ' husband and wife') #left space (otherwise e.g. woman and wife => wohusband and wife)
        #mankind
        s = s.replace(' mankind', ' humankind')     #left space (otherwise e.g. humankind => huhumankind)
        #man-made
        s = s.replace(' man-made', ' human-made')   #left space (otherwise e.g. human-made => huhuman-made)
        #manpower
        #s = s.replace('manpower', 'staff')  Depends on context
        #workmanlike
        s = s.replace('workmanlike', 'skillful')
        #workmanlike
        s = s.replace('freshman', 'first-year student')
        # #titles
        # s = s.replace('Mrs.', 'Ms.')
        # s = s.replace('Mrs ', 'Ms.')
        # s = s.replace('Miss ', 'Ms ')
        # s = s.replace('Miss. ', 'Ms. ')


        #####################
        #4. ALTERNATE ORDER #
        #####################
        #Depending on counts => men and women => women and men
        #OR? Make completely neutral => man/woman => person. boy/girl => kid, grandmother, grandfather => grandparent, godmother/godfather =>godparent....
        #Ms., Mr., Mrs. => Mx., brother/sister => sibling, husband/wife => spouse, man/woman => person/human?
        # list of all elements: https://genderrights.org.au/faq_type/language/
        #https://ielts.com.au/articles/grammar-101-feminine-and-masculine-words-in-english /
        new_sentences.append(s)
    return new_sentences
#python exploration_and_mining.py -user_task "Create a new folder in Markor named folder_20250731_222241." -task_dir 'MarkorCreateFolder' -package_name 'net.gsantner.markor'
#python exploration_and_mining.py -user_task "Create a new note in Markor named final_cool_fish.txt with the following text: Don't cry over spilled milk." -task_dir 'MarkorCreateNote' -package_name 'net.gsantner.markor'
#python exploration_and_mining.py -user_task "Create a note in Markor named 2023_05_15_fair_house.md. Perform a paste operation in the note and save the note." -task_dir 'MarkorCreateNoteFromClipboard' -package_name 'net.gsantner.markor'
#python exploration_and_mining.py -user_task "Delete all my notes in Markor." -task_dir 'MarkorDeleteAllNotes' -package_name 'net.gsantner.markor'
#python exploration_and_mining.py -user_task "Add a favorite location marker for 47.1858882, 9.5452201 in the OsmAnd maps app." -task_dir 'OsmAndFavorite' -package_name 'net.osmand'
#python exploration_and_mining.py -user_task "Add a location marker for 47.1026191, 9.6083057 in the OsmAnd maps app." -task_dir 'OsmAndMarker' -package_name 'net.osmand'
#python exploration_and_mining.py -user_task "Save a track with waypoints Balzers, Liechtenstein, Planken, Liechtenstein, Malbun, Liechtenstein in the OsmAnd maps app in the same order as listed." -task_dir 'OsmAndTrack' -package_name 'net.osmand'
#python exploration_and_mining.py -user_task """Recipe: Vegetable Stir Fry with Tofu
# description: A quick and easy meal, perfect for busy weekdays.
# servings: 1 serving
# preparationTime: 2 hrs
# ingredients: per individual taste
# directions: Stir-fry tofu cubes until golden, add assorted vegetables and a stir-fry sauce. Serve over rice or noodles. Try adding a pinch of your favorite spices for extra flavor.
#
#Recipe: Garlic Butter Shrimp
# description: A delicious and healthy choice for any time of the day.
# servings: 1 serving
# preparationTime: 20 mins
# ingredients: as per recipe
# directions: Sauté shrimp in butter and minced garlic until pink. Sprinkle with parsley and serve with lemon wedges. Garnish with fresh herbs for a more vibrant taste.
#
#Recipe: Lentil Soup
# description: A quick and easy meal, perfect for busy weekdays.
# servings: 6 servings
# preparationTime: 10 mins
# ingredients: as needed
# directions: Cook onions, carrots, celery, garlic, and lentils in vegetable broth until lentils are tender. Season with thyme and bay leaves. Feel free to substitute with ingredients you have on hand.""" -task_dir 'RecipeAddMultipleRecipes' -package_name 'com.flauschcode.broccoli'
#python exploration_and_mining.py -user_task "Delete the following recipes from Broccoli app: Lentil Soup and Garlic Butter Shrimp" -task_dir 'RecipeDeleteMultipleRecipes' -package_name 'com.flauschcode.broccoli'
#python exploration_and_mining.py -user_task "Delete all but one of any recipes in the Broccoli app that are exact duplicates, ensuring at least one instance of each unique recipe remains" -task_dir 'RecipeDeleteDuplicateRecipes' -package_name 'com.flauschcode.broccoli'
#python exploration_and_mining.py -user_task "In Simple Calendar Pro, create a calendar event on 2023-10-26 at 15h with the title 'Review session for Project X' and the description 'We will explore business objectives.'. The event should last for 45 mins." -task_dir 'SimpleCalendarAddOneEvent' -package_name 'com.simplemobiletools.calendar.pro'
#python exploration_and_mining.py -user_task "In Simple Calendar Pro, create a calendar event in two weeks from today at 16h with the title 'Call with Alice' and the description 'We will plan upcoming project milestones.'. The event should last for 15 mins." -task_dir 'SimpleCalendarAddOneEventInTwoWeeks' -package_name 'com.simplemobiletools.calendar.pro'
#python exploration_and_mining.py -user_task "In Simple Calendar Pro, create a calendar event for this Tuesday at 17h with the title 'Appointment for Project X' and the description 'We will review annual budget.'. The event should last for 30 mins." -task_dir 'SimpleCalendarAddOneEventRelativeDay' -package_name 'com.simplemobiletools.calendar.pro'
#python exploration_and_mining.py -user_task "In Simple Calendar Pro, create a calendar event for tomorrow at 12h with the title 'Review session for Budget Planning' and the description 'We will discuss business objectives. Let's be punctual.'. The event should last for 60 mins." -task_dir 'SimpleCalendarAddOneEventTomorrow' -package_name 'com.simplemobiletools.calendar.pro'
#python exploration_and_mining.py -user_task "In Simple Calendar Pro, create a recurring calendar event titled 'Review session for Annual Report' starting on 2023-10-18 at 17h. The event recurs weekly, forever, and lasts for 60 minutes each occurrence. The event description should be 'We will strategize about business objectives.'." -task_dir 'SimpleCalendarAddRepeatingEvent' -package_name 'com.simplemobiletools.calendar.pro'
#python exploration_and_mining.py -user_task "What events do I have October 26 2023 in Simple Calendar Pro? Answer with the titles only. If there are multiple titles, format your answer as a comma separated list.
#SimpleCalendarFirstEventAfterStartTime'." -task_dir 'SimpleCalendarEventsOnDate' -package_name 'com.simplemobiletools.calendar.pro'
#python exploration_and_mining.py -user_task "In Simple Calendar Pro, delete all the calendar events on 2023-10-26" -task_dir 'SimpleCalendarDeleteEvents' -package_name 'com.simplemobiletools.calendar.pro'
python exploration_and_mining.py -user_task "Create a new drawing in Simple Draw Pro. Name it amet_2023_01_26_super_umbrella.jpg. Save it in the Pictures folder within the sdk_gphone_x86_64 storage area." -task_dir 'SimpleDrawProCreateDrawing' -package_name 'com.simplemobiletools.draw.pro'
python exploration_and_mining.py -user_task "Reply to +15407572040 with message: Gym membership renewal due on the 20th. in Simple SMS Messenger" -task_dir 'SimpleSmsReply' -package_name 'com.simplemobiletools.smsmessenger'
python exploration_and_mining.py -user_task "Reply to the most recent text message using Simple SMS Messenger with message: Weekend plans: Hiking trip to Blue Mountain." -task_dir 'SimpleSmsReplyMostRecent' -package_name 'com.simplemobiletools.smsmessenger'
python exploration_and_mining.py -user_task "Send a message to +12959880189 with the clipboard content in Simple SMS Messenger" -task_dir 'SimpleSmsSendClipboardContent' -package_name 'com.simplemobiletools.smsmessenger'


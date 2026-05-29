TASK_LIST = [
    # "AudioRecorderRecordAudio",
    # "AudioRecorderRecordAudioWithFileName",
    # "BrowserDraw",
    # "BrowserMaze",
    # "BrowserMultiply",
    # "CameraTakePhoto",
    # "CameraTakeVideo",
    # "ClockStopWatchPausedVerify",
    # "ClockStopWatchRunning",
    "ClockTimerEntry",
    "ContactsAddContact",
    "ContactsNewContactDraft",
    "ExpenseAddMultiple",
    "ExpenseAddMultipleFromGallery",
    "ExpenseAddMultipleFromMarkor",
    "ExpenseAddSingle",
    "ExpenseDeleteDuplicates",
    "ExpenseDeleteDuplicates2",
    "ExpenseDeleteMultiple",
    "ExpenseDeleteMultiple2",
    "ExpenseDeleteSingle",
    "FilesDeleteFile",
    "FilesMoveFile",
    "MarkorAddNoteHeader",
    "MarkorChangeNoteContent",
    "MarkorCreateFolder",
    "MarkorCreateNote",
    "MarkorCreateNoteAndSms",
    "MarkorCreateNoteFromClipboard",
    "MarkorDeleteAllNotes",
    "MarkorDeleteNewestNote",
    "MarkorDeleteNote",
    "MarkorEditNote",
    "MarkorMergeNotes",
    "MarkorMoveNote",
    "MarkorTranscribeReceipt",
    "MarkorTranscribeVideo",
    "NotesIsTodo",
    "NotesMeetingAttendeeCount",
    "NotesRecipeIngredientCount",
    "NotesTodoItemCount",
    "OpenAppTaskEval",
    "OsmAndFavorite",
    "OsmAndMarker",
    "OsmAndTrack",
    "RecipeAddMultipleRecipes",
    "RecipeAddMultipleRecipesFromImage",
    "RecipeAddMultipleRecipesFromMarkor",
    "RecipeAddMultipleRecipesFromMarkor2",
    "RecipeAddSingleRecipe",
    "RecipeDeleteDuplicateRecipes",
    "RecipeDeleteDuplicateRecipes2",
    "RecipeDeleteDuplicateRecipes3",
    "RecipeDeleteMultipleRecipes",
    "RecipeDeleteMultipleRecipesWithConstraint",
    "RecipeDeleteMultipleRecipesWithNoise",
    "RecipeDeleteSingleRecipe",
    "RecipeDeleteSingleWithRecipeWithNoise",
    "RetroCreatePlaylist",
    "RetroPlayingQueue",
    "RetroPlaylistDuration",
    "RetroSavePlaylist",
    "SaveCopyOfReceiptTaskEval",
    "SimpleCalendarAddOneEvent",
    "SimpleCalendarAddOneEventInTwoWeeks",
    "SimpleCalendarAddOneEventRelativeDay",
    "SimpleCalendarAddOneEventTomorrow",
    "SimpleCalendarAddRepeatingEvent",
    "SimpleCalendarAnyEventsOnDate",
    "SimpleCalendarDeleteEvents",
    "SimpleCalendarDeleteEventsOnRelativeDay",
    "SimpleCalendarDeleteOneEvent",
    "SimpleCalendarEventOnDateAtTime",
    "SimpleCalendarEventsInNextWeek",
    "SimpleCalendarEventsInTimeRange",
    "SimpleCalendarEventsOnDate",
    "SimpleCalendarFirstEventAfterStartTime",
    "SimpleCalendarLocationOfEvent",
    "SimpleCalendarNextEvent",
    "SimpleCalendarNextMeetingWithPerson",
    "SimpleDrawProCreateDrawing",
    "SimpleSmsReply",
    "SimpleSmsReplyMostRecent",
    "SimpleSmsResend",
    "SimpleSmsSend",
    "SimpleSmsSendClipboardContent",
    "SimpleSmsSendReceivedAddress",
    "SportsTrackerActivitiesCountForWeek",
    "SportsTrackerActivitiesOnDate",
    "SportsTrackerActivityDuration",
    "SportsTrackerLongestDistanceActivity",
    "SportsTrackerTotalDistanceForCategoryOverInterval",
    "SportsTrackerTotalDurationForCategoryThisWeek",
    "SystemBluetoothTurnOff",
    "SystemBluetoothTurnOffVerify",
    "SystemBluetoothTurnOn",
    "SystemBluetoothTurnOnVerify",
    "SystemBrightnessMax",
    "SystemBrightnessMaxVerify",
    "SystemBrightnessMin",
    "SystemBrightnessMinVerify",
    "SystemCopyToClipboard",
    "SystemWifiTurnOff",
    "SystemWifiTurnOffVerify",
    "SystemWifiTurnOn",
    "SystemWifiTurnOnVerify",
    "TasksCompletedTasksForDate",
    "TasksDueNextWeek",
    "TasksDueOnDate",
    "TasksHighPriorityTasks",
    "TasksHighPriorityTasksDueOnDate",
    "TasksIncompleteTasksOnDate",
    "TurnOffWifiAndTurnOnBluetooth",
    "TurnOnWifiAndOpenApp",
    "VlcCreatePlaylist",
    "VlcCreateTwoPlaylists"
]

TASK_APP_MAPPING = {
    # Audio Recorder
    'AudioRecorderRecordAudio': ('com.dimowner.audiorecorder', 'audio recorder'),
    'AudioRecorderRecordAudioWithFileName': ('com.dimowner.audiorecorder', 'audio recorder'),

    # Browser (Chrome)
    'BrowserDraw': ('com.google.android.documentsui', 'files'),
    'BrowserMaze': ('com.google.android.documentsui', 'files'),
    'BrowserMultiply': ('com.google.android.documentsui', 'files'),

    # Calendar (Simple Calendar Pro)
    'SimpleCalendarAddOneEvent': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarAddOneEventInTwoWeeks': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarAddOneEventRelativeDay': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarAddOneEventTomorrow': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarAddRepeatingEvent': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarDeleteEvents': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarDeleteEventsOnRelativeDay': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarDeleteOneEvent': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarEventOnDateAtTime': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarEventsInNextWeek': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarEventsInTimeRange': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarEventsOnDate': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarFirstEventAfterStartTime': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarLocationOfEvent': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarNextEvent': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),
    'SimpleCalendarNextMeetingWithPerson': ('com.simplemobiletools.calendar.pro', 'simple calendar pro'),

    # Camera
    'CameraTakePhoto': ('com.android.camera2', 'camera'),
    'CameraTakeVideo': ('com.android.camera2', 'camera'),

    # Clock
    'ClockStopWatchPausedVerify': ('com.google.android.deskclock', 'clock'),
    'ClockStopWatchRunning': ('com.google.android.deskclock', 'clock'),
    'ClockTimerEntry': ('com.google.android.deskclock', 'clock'),

    # Contacts
    'ContactsAddContact': ('com.google.android.contacts', 'contacts'),
    'ContactsNewContactDraft': ('com.google.android.contacts', 'contacts'),

    # Pro Expense
    'ExpenseAddMultiple': ('com.arduia.expense', 'pro expense'),
    'ExpenseAddMultipleFromGallery': ('com.arduia.expense', 'pro expense'),
    'ExpenseAddMultipleFromMarkor': ('com.arduia.expense', 'pro expense'),
    'ExpenseAddSingle': ('com.arduia.expense', 'pro expense'),
    'ExpenseDeleteDuplicates': ('com.arduia.expense', 'pro expense'),
    'ExpenseDeleteDuplicates2': ('com.arduia.expense', 'pro expense'),
    'ExpenseDeleteMultiple': ('com.arduia.expense', 'pro expense'),
    'ExpenseDeleteMultiple2': ('com.arduia.expense', 'pro expense'),
    'ExpenseDeleteSingle': ('com.arduia.expense', 'pro expense'),

    # Files
    'FilesDeleteFile': ('com.google.android.documentsui', 'files'),
    'FilesMoveFile': ('com.google.android.documentsui', 'files'),

    # Markor
    'MarkorAddNoteHeader': ('net.gsantner.markor', 'markor'),
    'MarkorChangeNoteContent': ('net.gsantner.markor', 'markor'),
    'MarkorCreateFolder': ('net.gsantner.markor', 'markor'),
    'MarkorCreateNote': ('net.gsantner.markor', 'markor'),
    'MarkorCreateNoteFromClipboard': ('net.gsantner.markor', 'markor'),
    'MarkorDeleteAllNotes': ('net.gsantner.markor', 'markor'),
    'MarkorDeleteNewestNote': ('net.gsantner.markor', 'markor'),
    'MarkorDeleteNote': ('net.gsantner.markor', 'markor'),
    'MarkorEditNote': ('net.gsantner.markor', 'markor'),
    'MarkorMergeNotes': ('net.gsantner.markor', 'markor'),
    'MarkorMoveNote': ('net.gsantner.markor', 'markor'),
    'MarkorTranscribeReceipt': ('net.gsantner.markor', 'markor'),
    'MarkorTranscribeVideo': ('net.gsantner.markor', 'markor'),

    # Markor + SMS composite
    'MarkorCreateNoteAndSms': ('net.gsantner.markor', 'markor'),  # 复合任务，先用markor

    # information retrieval in joplin
    "NotesIsTodo" : ('net.cozic.joplin', 'joplin'),
    "NotesMeetingAttendeeCount": ('net.cozic.joplin', 'joplin'),
    "NotesRecipeIngredientCount": ('net.cozic.joplin', 'joplin'),
    "NotesTodoItemCount": ('net.cozic.joplin', 'joplin'),

    # OsmAnd
    'OsmAndFavorite': ('net.osmand', 'osmand'),
    'OsmAndMarker': ('net.osmand', 'osmand'),
    'OsmAndTrack': ('net.osmand', 'osmand'),

    # Recipe (Broccoli)
    'RecipeAddMultipleRecipes': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeAddMultipleRecipesFromImage': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeAddMultipleRecipesFromMarkor': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeAddMultipleRecipesFromMarkor2': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeAddSingleRecipe': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteDuplicateRecipes': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteDuplicateRecipes2': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteDuplicateRecipes3': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteMultipleRecipes': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteMultipleRecipesWithConstraint': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteMultipleRecipesWithNoise': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteSingleRecipe': ('com.flauschcode.broccoli', 'broccoli'),
    'RecipeDeleteSingleWithRecipeWithNoise': ('com.flauschcode.broccoli', 'broccoli'),

    # Retro Music
    'RetroCreatePlaylist': ('code.name.monkey.retromusic', 'retro music'),
    'RetroPlayingQueue': ('code.name.monkey.retromusic', 'retro music'),
    'RetroPlaylistDuration': ('code.name.monkey.retromusic', 'retro music'),
    'RetroSavePlaylist': ('code.name.monkey.retromusic', 'retro music'),

    # Simple Draw Pro
    'SimpleDrawProCreateDrawing': ('com.simplemobiletools.draw.pro', 'simple draw pro'),

    # Simple Gallery Pro
    'SaveCopyOfReceiptTaskEval': ('com.simplemobiletools.gallery.pro', 'simple gallery pro'),

    # SMS (Simple SMS Messenger)
    'SimpleSmsReply': ('com.simplemobiletools.smsmessenger', 'simple sms messenger'),
    'SimpleSmsReplyMostRecent': ('com.simplemobiletools.smsmessenger', 'simple sms messenger'),
    'SimpleSmsResend': ('com.simplemobiletools.smsmessenger', 'simple sms messenger'),
    'SimpleSmsSend': ('com.simplemobiletools.smsmessenger', 'simple sms messenger'),
    'SimpleSmsSendClipboardContent': ('com.simplemobiletools.smsmessenger', 'simple sms messenger'),
    'SimpleSmsSendReceivedAddress': ('com.simplemobiletools.smsmessenger', 'simple sms messenger'),

    #sport tracker
    "SportsTrackerActivitiesCountForWeek": ('de.dennisguse.opentracks', 'open tracks sports tracker'),
    "SportsTrackerActivitiesOnDate": ('de.dennisguse.opentracks', 'open tracks sports tracker'),
    "SportsTrackerActivityDuration": ('de.dennisguse.opentracks', 'open tracks sports tracker'),
    "SportsTrackerLongestDistanceActivity": ('de.dennisguse.opentracks', 'open tracks sports tracker'),
    "SportsTrackerTotalDistanceForCategoryOverInterval": ('de.dennisguse.opentracks', 'open tracks sports tracker'),
    "SportsTrackerTotalDurationForCategoryThisWeek": ('de.dennisguse.opentracks', 'open tracks sports tracker'),

    # System tasks (需要Settings应用)
    'OpenAppTaskEval': (None, None),  # 这个任务是打开其他应用，不需要settings
    'SystemBluetoothTurnOff': ('com.android.settings', 'settings'),
    'SystemBluetoothTurnOffVerify': ('com.android.settings', 'settings'),
    'SystemBluetoothTurnOn': ('com.android.settings', 'settings'),
    'SystemBluetoothTurnOnVerify': ('com.android.settings', 'settings'),
    'SystemBrightnessMax': ('com.android.settings', 'settings'),
    'SystemBrightnessMaxVerify': ('com.android.settings', 'settings'),
    'SystemBrightnessMin': ('com.android.settings', 'settings'),
    'SystemBrightnessMinVerify': ('com.android.settings', 'settings'),
    'SystemCopyToClipboard': (None, None),  # 剪贴板操作不需要特定应用
    'SystemWifiTurnOff': ('com.android.settings', 'settings'),
    'SystemWifiTurnOffVerify': ('com.android.settings', 'settings'),
    'SystemWifiTurnOn': ('com.android.settings', 'settings'),
    'SystemWifiTurnOnVerify': ('com.android.settings', 'settings'),

    # Task anwser
    "TasksCompletedTasksForDate": ('org.tasks', 'tasks'),
    "TasksDueNextWeek": ('org.tasks', 'tasks'),
    "TasksDueOnDate": ('org.tasks', 'tasks'),
    "TasksHighPriorityTasksDueOnDate": ('org.tasks', 'tasks'),
    "TasksIncompleteTasksOnDate": ('org.tasks', 'tasks'),

    # System composite tasks
    'TurnOffWifiAndTurnOnBluetooth': ('com.android.settings', 'settings'),
    'TurnOnWifiAndOpenApp': ('com.android.settings', 'settings'),

    # VLC
    'VlcCreatePlaylist': ('org.videolan.vlc', 'vlc'),
    'VlcCreateTwoPlaylists': ('org.videolan.vlc', 'vlc'),
}

PACKAGE_APP_MAPPING = {
    # Audio Recorder
    'com.dimowner.audiorecorder': 'audio-recorder',

    # Browser (Chrome)
    'com.google.android.documentsui': 'files',

    # Calendar (Simple Calendar Pro)
    'com.simplemobiletools.calendar.pro': 'simple-calendar-pro',

    # Camera
    'com.android.camera2': 'camera',

    # Clock
    'com.google.android.deskclock': 'clock',

    # Contacts
    'com.google.android.contacts': 'contacts',

    # Pro Expense
    'com.arduia.expense': 'pro-expense',


    # Markor
    'net.gsantner.markor': 'markor',


    # information retrieval in joplin
    'net.cozic.joplin': 'joplin',

    # OsmAnd
    'net.osmand': 'osmand',

    # Recipe (Broccoli)
    'com.flauschcode.broccoli': 'broccoli',

    # Retro Music
    'code.name.monkey.retromusic': 'retro-music',


    # Simple Draw Pro
    'com.simplemobiletools.draw.pro': 'simple-draw-pro',

    # Simple Gallery Pro
    'com.simplemobiletools.gallery.pro': 'simple-gallery-pro',

    # SMS (Simple SMS Messenger)
    'com.simplemobiletools.smsmessenger': 'simple-sms-messenger',


    #sport tracker
    'de.dennisguse.opentracks': 'open-tracks-sports-tracker',

    # System tasks (需要Settings应用)

    'com.android.settings': 'settings',


    # Task anwser
    'org.tasks': 'tasks',


    # VLC
    'org.videolan.vlc': 'vlc',
}


def get_app_info(task_name: str):
    """获取任务对应的应用信息"""
    return TASK_APP_MAPPING.get(task_name, (None, None))
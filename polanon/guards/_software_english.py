"""The English of software: words of menus, buttons, settings and manuals, which name nobody.

An AI agent's prompt, tools and results are full of them, written with a capital letter as a
button or a heading writes them, and the Polish language model takes such a word for a name:
"Uncheck", "Authorization", "Privacy Policy". The list is written for this library, not taken
from a dictionary. A word that is also a first name or a surname is left out on purpose, such
as Price, Grant, Page, Key, Field, Post, Root, Banner, Story, Mark, Bill or Will, so a person
written that way is still a person.
"""

from __future__ import annotations

_WORDS = """
    accept access account accounts action actions activate active activity add additional admin
    administrator advanced agent alert alerts all allow analytics api app application apps apply
    approve archive assign assistant attach attachment attachments audit authentication author
    authority authorization authorize authors auto automatic automation availability available
    backup balance basic battery beta billing blank block board bookmark bookmarks branch browse
    browser budget bug bugs build bulk button cache calendar camera campaign cancel cart catalog
    categories category change changelog changes channel channels charging chart chat check
    checklist checkout choose clear click client clients clone close cloud code collapse
    collection collections collisions column columns command commands comment comments commit
    community compare compliance component components config configuration configure confirm
    connect connection console contact contacts content context continue contributors controls
    cookie cookies copy copyright core cost costs coverage create credentials credits current
    custom customer customers customize daily dashboard data database deactivate debug decline
    default defaults delete delivery demo deny deploy deployment deprecated description design
    desktop detailed details developer developers device devices dialog directory disable
    disabled disconnect discount discussion dismiss display docs document documentation
    documents domain download downloads draft drafts drag driver drop duplicate edit edition
    editor email emails enable enabled endpoint energy engine enter enterprise entities entity
    entry environment error errors event events example examples exception expand experimental
    export extension extensions external faq feature features feed feedback fetch file files
    filter filters finish folder folders follow footer form forms forum forward framework full
    function functions gallery gateway general generate global glossary group groups guide
    guidelines guides hardware header health help hidden hide history home honest hosting hub
    icon icons image images import inactive inbox index info information input insights install
    installation integration integrations interface internal introduction inventory invoice
    invoices issue issues item items join label labels language languages latest launch layout
    leave legacy legal level library license licenses limit limits link links list lists live
    load local lock log login logout logs main maintenance manage manager manual map maps
    marketplace media member members memory menu merge message messages metadata metrics
    migration minimize mobile mode model models module modules monitor monitoring monthly move
    mute name native network new news newsletter next node note notes notification
    notifications offer offers office official offline old online open option optional options
    order orders output overview owner package packages panel parameter parameters partner
    partners password passwords paste path pause payment payments performance permission
    permissions personal phone photo photos pin pitfalls platform play player plugin plugins
    policy popular portal preferences premium preview primary print privacy private product
    products profile program project projects properties property protocol provider proxy
    public publish pull purchase push query queue quick quickstart quota rating ratings readme
    recent recipe recommended record records refresh region register registry reject related
    release releases reload reminder reminders remote remove rename reply report reports
    repository request requests required requirements reset resize resource resources response
    restart restore result results resume retry return reversible revert review reviews role
    roles route rule rules run runtime sample samples sandbox save scan scene scenes schedule
    scheduling schema scope screen script scripts scroll search secondary section secure
    security select send sensor sensors server servers service services session sessions set
    settings setup share shared shipping shop shortcut shortcuts show sidebar sign signup simple
    site sitemap size skill skills skip smart snapshot software solution solutions sort source
    spam specs stable stack standard start statistics stats status step steps stop storage
    store stream studio submit subscribe subscription success summary support switch sync
    system systems tab table tables tag tags task tasks team teams template templates terminal
    terms test tests text theme themes thread threads ticket tickets timeline tips title toggle
    token tokens tool tools topic topics total tracker traffic training transaction transactions
    translate trash trial troubleshooting tutorial tutorials type types uncheck undo uninstall
    unit units unknown unlock unmute unpin unread unsubscribe unavailable update updates upgrade
    upload usage user users utilities validate value values variable variables vendor verified
    verify version versions video videos view views visible volume wallet warning warnings
    weather webhook webhooks website weekly widget widgets wiki window wizard workflow workflows
    workspace yearly zone zones zoom
"""

#: The words, in lowercase.
SOFTWARE_WORDS = frozenset(_WORDS.split())

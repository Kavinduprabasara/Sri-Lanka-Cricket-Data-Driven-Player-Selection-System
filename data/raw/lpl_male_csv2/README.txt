All Men's Lanka Premier League match data in CSV format
=======================================================

The background
--------------

As an experiment, after being asked by a user of the site, I started
converting the YAML data provided on the site into a CSV format. That
initial version was heavily influenced by the format used by the baseball
project Retrosheet. I wasn't sure of the usefulness of my CSV format, but
nothing better was suggested so I persisted with it. Later Ashwin Raman
(https://twitter.com/AshwinRaman_) send me a detailed example of a format
he felt might work and, liking what I saw, I started to produce data in
a slightly modified version of that initial example.

This particular zip folder contains the CSV data for...
  All Men's Lanka Premier League matches
...for which we have data.

How you can help
----------------

Providing feedback on the data would be the most helpful. Tell me what you
like and what you don't. Is there anything that is in the JSON data that
you'd like to be included in the CSV? Could something be included in a better
format? General views and comments help, as well as incredibly detailed
feedback. All information is of use to me at this stage. I can only improve
the data if people tell me what does works and what doesn't. I'd like to make
the data as useful as possible but I need your help to do it. Also, which of
the 2 CSV formats do you prefer, this one or the original? Ideally I'd like
to settle on a single CSV format so what should be kept from each?

Finally, any feedback as to the licence the data should be released under
would be greatly appreciated. Licensing is a strange little world and I'd
like to choose the "right" licence. My basic criteria may be that:

  * the data should be free,
  * corrections are encouraged/required to be reported to the project,
  * derivative works are allowed,
  * you can't just take data and sell it.

Feedback, pointers, comments, etc on licensing are welcome.

The format of the data
----------------------

Full documentation of this CSV format can be found at:
  https://cricsheet.org/format/csv_ashwin/
but the following is a brief summary of the details...

This format consists of 2 files per match, although you can get all of
the ball-by-ball data from just one of the files. The files for a match
are named <id>.csv (for the ball-by-ball data), and <id>_info.csv (for
the match info), where <id> is the Cricinfo id for the match. The
ball-by-ball file contains one row per delivery in the match, while the
match info file contains match information such as dates the match was
played, the outcome, and lists of the players involved in the match.

The match info file format
--------------------------

The info section contains the information on the actual match, such as
when and where it was played, any event it was part of, the type of
match etc. The fields included in the info section will each appear as
one or more rows in the data. Some of the fields are required, whereas
some are optional. If a field has multiple values, such as team, then
each value will appear on a row of it's own.

The ball-by-ball file format
----------------------------

The first row of each ball-by-ball CSV file contains the headers for the
file, with each subsequent row providing details on a single delivery.
The headers in the file are:

  * match_id
  * season
  * start_date
  * venue
  * innings
  * ball
  * batting_team
  * bowling_team
  * striker
  * non_striker
  * bowler
  * runs_off_bat
  * extras
  * wides
  * noballs
  * byes
  * legbyes
  * penalty
  * wicket_type
  * player_dismissed
  * other_wicket_type
  * other_player_dismissed

Most of the fields above should, hopefully, be self-explanatory, but some may
benefit from clarification...

"innings" contains the number of the innings within the match. If a match is
one that would normally have 2 innings, such as a T20 or ODI, then any innings
of more than 2 can be regarded as a super over.

"ball" is a combination of the over and delivery. For example, "0.3" represents
the 3rd ball of the 1st over.

"wides", "noballs", "byes", "legbyes", and "penalty" contain the total of each
particular type of extras, or are blank if not relevant to the delivery.

If a wicket occurred on a delivery then "wicket_type" will contain the method
of dismissal, while "player_dismissed" will indicate who was dismissed. There
is also the, admittedly remote, possibility that a second dismissal can be
recorded on the delivery (such as when a player retires on the same delivery
as another dismissal occurs). In this case "other_wicket_type" will record
the reason, while "other_player_dismissed" will show who was dismissed.

Matches included in this archive
--------------------------------

2024-07-21 - club - LPL - male - 1428482 - Galle Marvels vs Jaffna Kings
2024-07-20 - club - LPL - male - 1428481 - Jaffna Kings vs Kandy Falcons
2024-07-18 - club - LPL - male - 1428480 - Colombo Strikers vs Kandy Falcons
2024-07-18 - club - LPL - male - 1428479 - Jaffna Kings vs Galle Marvels
2024-07-16 - club - LPL - male - 1428478 - Dambulla Sixers vs Colombo Strikers
2024-07-15 - club - LPL - male - 1428477 - Galle Marvels vs Colombo Strikers
2024-07-15 - club - LPL - male - 1428476 - Kandy Falcons vs Dambulla Sixers
2024-07-14 - club - LPL - male - 1428475 - Jaffna Kings vs Colombo Strikers
2024-07-14 - club - LPL - male - 1428474 - Galle Marvels vs Dambulla Sixers
2024-07-13 - club - LPL - male - 1428473 - Kandy Falcons vs Jaffna Kings
2024-07-10 - club - LPL - male - 1428472 - Kandy Falcons vs Galle Marvels
2024-07-10 - club - LPL - male - 1428471 - Colombo Strikers vs Jaffna Kings
2024-07-09 - club - LPL - male - 1428470 - Dambulla Sixers vs Galle Marvels
2024-07-09 - club - LPL - male - 1428469 - Jaffna Kings vs Kandy Falcons
2024-07-07 - club - LPL - male - 1428468 - Colombo Strikers vs Dambulla Sixers
2024-07-07 - club - LPL - male - 1428467 - Kandy Falcons vs Galle Marvels
2024-07-06 - club - LPL - male - 1428466 - Jaffna Kings vs Dambulla Sixers
2024-07-06 - club - LPL - male - 1428465 - Colombo Strikers vs Kandy Falcons
2024-07-05 - club - LPL - male - 1428464 - Galle Marvels vs Jaffna Kings
2024-07-03 - club - LPL - male - 1428463 - Galle Marvels vs Colombo Strikers
2024-07-03 - club - LPL - male - 1428462 - Dambulla Sixers vs Jaffna Kings
2024-07-02 - club - LPL - male - 1428461 - Colombo Strikers vs Kandy Falcons
2024-07-02 - club - LPL - male - 1428460 - Jaffna Kings vs Galle Marvels
2024-07-01 - club - LPL - male - 1428459 - Dambulla Sixers vs Kandy Falcons
2023-08-20 - club - LPL - male - 1382901 - Dambulla Aura vs B-Love Kandy
2023-08-19 - club - LPL - male - 1382900 - B-Love Kandy vs Galle Titans
2023-08-17 - club - LPL - male - 1382899 - B-Love Kandy vs Jaffna Kings
2023-08-17 - club - LPL - male - 1382898 - Galle Titans vs Dambulla Aura
2023-08-15 - club - LPL - male - 1382897 - Colombo Strikers vs Galle Titans
2023-08-14 - club - LPL - male - 1382896 - Dambulla Aura vs B-Love Kandy
2023-08-13 - club - LPL - male - 1382895 - Colombo Strikers vs B-Love Kandy
2023-08-13 - club - LPL - male - 1382894 - Jaffna Kings vs Galle Titans
2023-08-12 - club - LPL - male - 1382893 - Dambulla Aura vs Colombo Strikers
2023-08-12 - club - LPL - male - 1382892 - B-Love Kandy vs Jaffna Kings
2023-08-11 - club - LPL - male - 1382891 - Galle Titans vs Dambulla Aura
2023-08-08 - club - LPL - male - 1382890 - Colombo Strikers vs Jaffna Kings
2023-08-08 - club - LPL - male - 1382889 - B-Love Kandy vs Galle Titans
2023-08-07 - club - LPL - male - 1382888 - Dambulla Aura vs Jaffna Kings
2023-08-07 - club - LPL - male - 1382887 - Galle Titans vs Colombo Strikers
2023-08-05 - club - LPL - male - 1382886 - Jaffna Kings vs B-Love Kandy
2023-08-05 - club - LPL - male - 1382885 - Dambulla Aura vs Colombo Strikers
2023-08-04 - club - LPL - male - 1382884 - Galle Titans vs Jaffna Kings
2023-08-04 - club - LPL - male - 1382883 - Dambulla Aura vs B-Love Kandy
2023-08-01 - club - LPL - male - 1382882 - Galle Titans vs B-Love Kandy
2023-08-01 - club - LPL - male - 1382881 - Jaffna Kings vs Dambulla Aura
2023-07-31 - club - LPL - male - 1382880 - Colombo Strikers vs B-Love Kandy
2023-07-31 - club - LPL - male - 1382879 - Galle Titans vs Dambulla Aura
2023-07-30 - club - LPL - male - 1382878 - Jaffna Kings vs Colombo Strikers
2022-12-23 - club - LPL - male - 1324542 - Colombo Stars vs Jaffna Kings
2022-12-22 - club - LPL - male - 1324541 - Kandy Falcons vs Colombo Stars
2022-12-21 - club - LPL - male - 1324540 - Galle Gladiators vs Colombo Stars
2022-12-21 - club - LPL - male - 1324539 - Kandy Falcons vs Jaffna Kings
2022-12-19 - club - LPL - male - 1324538 - Galle Gladiators vs Dambulla Aura
2022-12-19 - club - LPL - male - 1324537 - Colombo Stars vs Jaffna Kings
2022-12-18 - club - LPL - male - 1324536 - Kandy Falcons vs Dambulla Aura
2022-12-18 - club - LPL - male - 1324535 - Jaffna Kings vs Galle Gladiators
2022-12-17 - club - LPL - male - 1324534 - Colombo Stars vs Kandy Falcons
2022-12-17 - club - LPL - male - 1324533 - Dambulla Aura vs Galle Gladiators
2022-12-14 - club - LPL - male - 1324532 - Dambulla Aura vs Colombo Stars
2022-12-14 - club - LPL - male - 1324531 - Kandy Falcons vs Jaffna Kings
2022-12-13 - club - LPL - male - 1324530 - Galle Gladiators vs Colombo Stars
2022-12-13 - club - LPL - male - 1324529 - Kandy Falcons vs Dambulla Aura
2022-12-12 - club - LPL - male - 1324528 - Jaffna Kings vs Colombo Stars
2022-12-12 - club - LPL - male - 1324527 - Galle Gladiators vs Kandy Falcons
2022-12-11 - club - LPL - male - 1324526 - Jaffna Kings vs Dambulla Aura
2022-12-11 - club - LPL - male - 1324525 - Galle Gladiators vs Colombo Stars
2022-12-10 - club - LPL - male - 1324524 - Jaffna Kings vs Kandy Falcons
2022-12-08 - club - LPL - male - 1324523 - Colombo Stars vs Dambulla Aura
2022-12-07 - club - LPL - male - 1324522 - Galle Gladiators vs Kandy Falcons
2022-12-07 - club - LPL - male - 1324521 - Dambulla Aura vs Jaffna Kings
2022-12-06 - club - LPL - male - 1324520 - Kandy Falcons vs Colombo Stars
2022-12-06 - club - LPL - male - 1324519 - Jaffna Kings vs Galle Gladiators
2021-12-23 - club - LPL - male - 1291218 - Jaffna Kings vs Galle Gladiators
2021-12-21 - club - LPL - male - 1291217 - Jaffna Kings vs Dambulla Giants
2021-12-19 - club - LPL - male - 1291216 - Galle Gladiators vs Jaffna Kings
2021-12-19 - club - LPL - male - 1291215 - Colombo Stars vs Dambulla Giants
2021-12-17 - club - LPL - male - 1291214 - Colombo Stars vs Kandy Warriors
2021-12-17 - club - LPL - male - 1291213 - Galle Gladiators vs Jaffna Kings
2021-12-16 - club - LPL - male - 1291212 - Dambulla Giants vs Kandy Warriors
2021-12-16 - club - LPL - male - 1291211 - Jaffna Kings vs Colombo Stars
2021-12-14 - club - LPL - male - 1291210 - Galle Gladiators vs Dambulla Giants
2021-12-14 - club - LPL - male - 1291209 - Kandy Warriors vs Colombo Stars
2021-12-13 - club - LPL - male - 1291208 - Dambulla Giants vs Jaffna Kings
2021-12-12 - club - LPL - male - 1291207 - Colombo Stars vs Galle Gladiators
2021-12-12 - club - LPL - male - 1291206 - Kandy Warriors vs Jaffna Kings
2021-12-11 - club - LPL - male - 1291205 - Dambulla Giants vs Colombo Stars
2021-12-11 - club - LPL - male - 1291204 - Galle Gladiators vs Kandy Warriors
2021-12-10 - club - LPL - male - 1291203 - Jaffna Kings vs Colombo Stars
2021-12-10 - club - LPL - male - 1291202 - Dambulla Giants vs Galle Gladiators
2021-12-08 - club - LPL - male - 1291201 - Jaffna Kings vs Kandy Warriors
2021-12-08 - club - LPL - male - 1291200 - Dambulla Giants vs Colombo Stars
2021-12-07 - club - LPL - male - 1291199 - Kandy Warriors vs Galle Gladiators
2021-12-07 - club - LPL - male - 1291198 - Dambulla Giants vs Jaffna Kings
2021-12-06 - club - LPL - male - 1291197 - Galle Gladiators vs Colombo Stars
2021-12-06 - club - LPL - male - 1291196 - Dambulla Giants vs Kandy Warriors
2021-12-05 - club - LPL - male - 1291195 - Galle Gladiators vs Jaffna Kings
2020-12-16 - club - LPL - male - 1238774 - Jaffna Stallions vs Galle Gladiators
2020-12-14 - club - LPL - male - 1238773 - Jaffna Stallions vs Dambulla Viiking
2020-12-13 - club - LPL - male - 1238772 - Colombo Kings vs Galle Gladiators
2020-12-11 - club - LPL - male - 1238771 - Dambulla Viiking vs Colombo Kings
2020-12-10 - club - LPL - male - 1238770 - Kandy Tuskers vs Galle Gladiators
2020-12-10 - club - LPL - male - 1238769 - Colombo Kings vs Jaffna Stallions
2020-12-09 - club - LPL - male - 1238768 - Galle Gladiators vs Dambulla Viiking
2020-12-09 - club - LPL - male - 1238767 - Jaffna Stallions vs Kandy Tuskers
2020-12-07 - club - LPL - male - 1238766 - Dambulla Viiking vs Jaffna Stallions
2020-12-07 - club - LPL - male - 1238765 - Colombo Kings vs Galle Gladiators
2020-12-05 - club - LPL - male - 1238764 - Kandy Tuskers vs Colombo Kings
2020-12-05 - club - LPL - male - 1238763 - Dambulla Viiking vs Galle Gladiators
2020-12-04 - club - LPL - male - 1238762 - Jaffna Stallions vs Colombo Kings
2020-12-03 - club - LPL - male - 1238761 - Kandy Tuskers vs Dambulla Viiking
2020-12-03 - club - LPL - male - 1238760 - Galle Gladiators vs Jaffna Stallions
2020-12-01 - club - LPL - male - 1238759 - Jaffna Stallions vs Kandy Tuskers
2020-12-01 - club - LPL - male - 1238758 - Dambulla Viiking vs Colombo Kings
2020-11-30 - club - LPL - male - 1238757 - Kandy Tuskers vs Galle Gladiators
2020-11-30 - club - LPL - male - 1238756 - Jaffna Stallions vs Dambulla Viiking
2020-11-28 - club - LPL - male - 1238755 - Colombo Kings vs Galle Gladiators
2020-11-28 - club - LPL - male - 1238754 - Dambulla Viiking vs Kandy Tuskers
2020-11-27 - club - LPL - male - 1238753 - Galle Gladiators vs Jaffna Stallions
2020-11-26 - club - LPL - male - 1238752 - Kandy Tuskers vs Colombo Kings

Further information
-------------------

You can find all of our currently available data at https://cricsheet.org/

You can contact me via the following methods:
  Email   : stephen@cricsheet.org
  Mastodon: @cricsheet@deeden.co.uk

from datetime import datetime, timedelta

# 创建日历内容
ics_content = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//hacksw/handcal//NONSGML v1.0//EN
CALSCALE:GREGORIAN
X-WR-TIMEZONE:Europe/Copenhagen
"""

# 每日固定活动列表（替换为英文）
daily_events = [
    ("Exercise", "08:00", "09:00", "Keep Fitness"),
    ("Individual Study", "09:00", "11:00", "Individual Study"),
    ("iOS Programming", "11:00", "12:30", "Individual Study"),
    ("Cooking and Eating", "12:30", "13:30", "Routine"),
    ("English Learning", "13:30", "14:30", "Individual Study"),
    ("Individual Study", "14:30", "16:30", "Individual Study"),
    ("Rest and Entertainment", "16:30", "17:30", "Routine"),
    ("Dinner and Rest", "17:30", "18:30", "Routine"),
    ("iOS Programming", "18:30", "20:30", "Individual Study"),
    ("English Learning", "20:30", "21:30", "Individual Study"),
    ("Review", "21:30", "22:30", "Individual Study"),
    ("Personal Hygiene", "22:30", "23:00", "Routine"),
]

# 起始日期
start_date = datetime(2024, 5, 24)

# 时区信息
timezone = "Europe/Copenhagen"

# 生成每日活动的日程
for day in range(13):
    current_date = start_date + timedelta(days=day)
    for event in daily_events:
        event_start = datetime.strptime(f"{current_date.strftime('%Y-%m-%d')} {event[1]}", "%Y-%m-%d %H:%M")
        event_end = datetime.strptime(f"{current_date.strftime('%Y-%m-%d')} {event[2]}", "%Y-%m-%d %H:%M")
        ics_content += "BEGIN:VEVENT\n"
        ics_content += f"UID:{event[0]}-{current_date.strftime('%Y%m%d')}\n"
        ics_content += f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}\n"
        ics_content += f"DTSTART;TZID={timezone}:{event_start.strftime('%Y%m%dT%H%M%S')}\n"
        ics_content += f"DTEND;TZID={timezone}:{event_end.strftime('%Y%m%dT%H%M%S')}\n"
        ics_content += f"SUMMARY:{event[0]}\n"
        ics_content += f"CATEGORIES:{event[3]}\n"
        ics_content += "RRULE:FREQ=DAILY\n"  # 设置为每天重复
        ics_content += "END:VEVENT\n"

# 结束日历文件
ics_content += "END:VCALENDAR\n"

# 将日历保存为 .ics 文件
with open("personal_schedule_with_categories.ics", "w") as file:
    file.write(ics_content)

print("ICS file has been generated: personal_schedule_with_categories.ics")

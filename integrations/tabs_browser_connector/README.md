# Tabs minutes comparison

This Chrome extension reads the **displayed** TabsNBA projections and chosen If/Then scenario.
The prop app compares those minutes with your own approved minutes. It never edits either site's
minutes, projections, prices, rankings or saved plays.

## Connection

1. After approving installation, open Chrome's Extensions page, enable Developer mode, and
   choose **Load unpacked** for this folder.
2. Reload the TabsNBA projections page and the prop app so Chrome can attach the connector.
3. On the Tabs projections tab you want to use, click **Tabs minutes comparison** in Chrome's
   Extensions menu. Its badge changes to **ON**. This explicitly selects one tab, so another
   Tabs window cannot overwrite your scenario. Reconnect after restarting Chrome.
4. Keep TabsNBA and `https://cktest2.streamlit.app/` open in the same Chrome profile. In
   Sportsbook Edges, choose **TabsNBA → Browser connection**. Choose **All games**, clear
   player/team/position filters and turn off Showdown CPT in Tabs for full-slate coverage.

The model checks the displayed source every 30 seconds. Tabs itself determines when its
projections update; this does not force a refresh, click any controls or bypass membership.
Only changed comparisons refresh the app's flags. Missing players remain unchecked. Only an
explicitly displayed OUT row becomes a zero-minute comparison. Its source/scenario remains
visible; it does not become an injury override in your model.

Personal minute/boost edits on Tabs suspend the independent browser comparison. You can
instead export a CSV and use **CSV upload**, which reads the separate **Tabs Min** column.
CSV imports require the selected scenario and visible publication time and do not auto-refresh.
Publication timestamps over two hours old cannot clear the minutes check. Closing the connected
Tabs tab, changing away from projections, or losing access clears the comparison on the next check.

## Access and privacy

- Access is limited to `app.tabsnba.com` and `cktest2.streamlit.app`; actual readers run on
  `/projections` and the prop app, respectively. No all-sites, cookies, browsing history,
  downloads or web-request permissions.
- Reads visible names, team/opponent, minutes, OUT labels, scenario, slate date and update time.
  No login details, hidden application state or authentication tokens are read.
- No external fetches. Chrome relays the data to your own open prop-app tab; Streamlit receives
  it in that tab's private session. Nothing is published to GitHub or shared with other visitors.
- The extension stores only the selected source tab ID, in Chrome session storage. No paid
  projection dataset is saved by the extension. Disable/remove the extension to disconnect.

After installation, check the connection status in Sportsbook Edges. Before Tabs publishes a slate,
the comparison stays unchecked. Automated tests cover parsing,
timestamps, identity matching, stale/edited data, missing rows, message routing and avoiding
dashboard reruns on unchanged polls. The prop app's embedded frame is supported within the same
approved `cktest2.streamlit.app` origin; frames on other sites cannot request the comparison.

Extract backup data from Label Studio via API:

```sh
curl -X GET 'http://localhost:8080/api/projects/<id>/export?exportType=JSON&download_all_tasks=true' -H 'Authorization: Token <token>' > backup_name.json
```

This <token> can be found in you Label Studio user settings under the "API Token" section.
I am using Legacy Token for authentication, so you also need to go to the "Organization" settings and enable "Legacy Token Authentication".

# Git Summary

## **1. Một số khái niệm**

### **Các thuật ngữ cơ bản**

- **Repository**: là nơi lưu trữ mã nguồn của dự án. Có 2 loại repository:
  - **Local repository**: là nơi lưu trữ source của dự án trên máy tính cá nhân.
  - **Remote repository**: là nơi lưu trữ mã nguồn của dự án trên máy chủ từ xa (remote server).

- **Commit**: là một bản ghi lịch sử của mã nguồn. Mỗi commit có một mã duy nhất (hash) để phân biệt với các commit khác. Mỗi commit có thể chứa nhiều file thay đổi. Mỗi commit có thể có một commit cha (parent commit) hoặc nhiều commit cha (parent commits) tùy thuộc vào số lượng branch được merge vào commit đó.

- **Branch**: là một nhánh của dự án. Mỗi branch có một tên duy nhất. Mỗi branch có thể có một branch cha (parent branch) hoặc nhiều branch cha (parent branches) tùy thuộc vào số lượng branch được merge vào branch đó.

- **Master/Main branch**: là branch mặc định của dự án. Mọi thay đổi của dự án đều được thực hiện trên master branch. Mỗi commit trên master branch đều có một commit cha (parent commit) là commit trước đó trên master branch.

- **Origin**: là tên đặt cho remote repository mặc định khi clone một remote repository.

- **HEAD**: là con trỏ trỏ tới commit hiện tại trên branch hiện tại.

- **Working directory**: là thư mục chứa mã nguồn của dự án.

- **Staging area**: là nơi lưu trữ các thay đổi của dự án trước khi commit. Staging Area nghĩa là khu vực sẽ lưu trữ những thay đổi của bạn trên tập tin để nó có thể được commit, vì muốn commit tập tin nào thì tập tin đó phải nằm trong Staging Area. Một tập tin khi nằm trong Staging Area sẽ có trạng thái là Stagged

### **Các trạng thái của file**

- **Untracked**: file chưa được thêm vào Staging Area.
- **Staged**: file đã được thêm vào Staging Area.
- **Unmodified**: là trạng thái của file khi nó chưa được chỉnh sửa so với commit trước đó.
- **Modified**: là trạng thái của một file khi nó đã được chỉnh sửa so với commit trước đó.
- **Tracked**: là trạng thái của file khi nó đã được thêm vào Git repository.

![git-workflow](https://git-scm.com/book/en/v2/images/lifecycle.png)

## **2. Git Config**

```bash
git config --global user.name "John Doe"
git config --global user.email
git config --list
git config --global --edit
```

- `--global` được dùng để config gloabal cho tất cả repo trong hệ thống. Nếu một config cho một repo cụ thể -> xóa  `--global`.
- `user.name` và `user.email` dùng để set tên và email của user.
- `git config --list` is used to list all the config set for the current repository.
- `git config --global --edit` is used to edit the config set for the current repository.

## **3. Git Init and Clone**

### **Git Init**

```bash
git init
```

- `git init` tạo một git repo tại thư mục hiện tại.

### **Git Clone**

```bash
git clone <url>
git clone ssh://<url>
```

- `git clone` dùng để sao chép một repo từ xa về máy tính.
- Có thể clone repo bằng `https` hoặc `ssh`.

## **4. Git status**

```bash
git status
```

- `git status` dùng để kiểm tra trạng thái của repo hiện tại. Các trạng thái có thể là `untracked`, `modified`, `staged`, `unmodified` đã nói ở trên.

## **5. Git Add**

- `git add -A` tương đương `git add -all`
- `git add - u` tương đương `git add --update`

| Command | New Files | Modified Files | Deleted Files| Description |
|:--|:--|:--|:--|:--|
|`git add -A`|✔️|✔️|✔️|Stage all (new, modified, deleted) files|
|`git add .`|✔️| ✔️|✔️|Stage all (new, modified, deleted) files **in current folder**|
|`git add --ignore-removal .`|✔️|✔️|❌ |Stage new and modified files only|
|`git add -u`|❌|✔️|✔️|Stage modified and deleted files only|

## **6. Git Commit**

```bash
git commit -m "Commit message"
```

- Dùng để commit các file đã được add vào local repo.
- Sửa commit message bằng `git commit --amend -m "New commit message"`
- Xoá commit bằng `git reset --hard HEAD^`
- Gộp commit bằng `git rebase -i HEAD~n` với n là số commit cần gộp. Sau đó chọn `squash` ở các commit cần gộp.
- `git commit --amend -m "nội dung commit message" --author="user.name <user.email>"` để thay đổi tác giả của commit.

### **Commit convention**

```md
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

- Trong đó:
  - type: từ khóa phân loại commit là feature, fix bug, refactor.
  - scope: trả lời câu hỏi: commit này refactor|fix cái gì? được đặt trong cặp ngoặc đơn ngay sau type. VD: feat(authentication):, fix(parser):
  - description: là mô tả ngắn về những gì sẽ bị sửa đổi trong commit đấy
  - body: là mô tả dài và chi tiết hơn, cần thiết khi description chưa thể nói rõ hết được
  - footer: một số thông tin mở rộng như số ID của pull request, issue..

- Một số type phổ biến được khuyên sử dụng bao gồm:
  - feat: thêm một feature
  - fix: fix bug cho hệ thống, vá lỗi trong codebase
  - refactor: sửa code nhưng không fix bug cũng không thêm feature hoặc đôi khi bug cũng được fix từ việc refactor.
  - docs: thêm/thay đổi document
  - chore: những sửa đổi nhỏ nhặt không liên quan tới code
  - style: những thay đổi không làm thay đổi ý nghĩa của code như thay đổi css/ui chẳng hạn.
  - perf: code cải tiến về mặt hiệu năng xử lý
  - vendor: cập nhật version cho các dependencies, packages.

## **7. Git Reset**

```bash
git reset HEAD
git reset HEAD^
git reset HEAD~n
git reset --soft HEAD^
git reset --hard HEAD^
git reset --mixed HEAD^
```

- `git reset HEAD` dùng để unstage các file đã được add vào staging area.
- `git reset HEAD^` dùng để uncommit commit cuối cùng.
- `git reset HEAD~n` quay trở lại n commit trước đó.

|| `--soft`| `--mixed`| `--hard` |
|:--|:--|:--|:--|
| HEAD| Di chuyển| Di chuyển| Di chuyển|
| Files| Không thay đổi| Được phục hồi theo commit| Được phục hồi theo commit|
| Staging Area | Không thay đổi| Được reset| Được reset|
| Commit History | Được giữ nguyên| Được giữ nguyên| Bị xóa từ HEAD trở đi|

![git-reset](https://i.stack.imgur.com/qRAte.jpg)

## **8. Git Revert**

```bash
git revert commit_hash
```

- `git revert commit_hash` dùng để tạo một commit mới chứa các thay đổi của commit_hash và đặt commit mới này lên commit_hash.

- Thay vì thực sự xóa commit đó đi như git reset --hard, git revert tạo ra 1 commit mới để hoàn tác lại các thay đổi trong commit đó.

- Nó sẽ tạo ra 1 commit mới với những thay đổi ngược lại để mang files trở về trạng thái trước khi commit đó được thêm vào. Do đó, commit bị revert vẫn còn trong lịch sử, nhưng ảnh hưởng của nó đã bị hoàn tác.

## **9. Git diff**

```bash
git diff
git diff --staged
git diff HEAD
```

- `git diff` so sánh sự khác nhau giữa working directory và staging area.
- `git diff --staged` so sánh sự khác nhau giữa staging area và local repository.
- `git diff HEAD` so sánh sự khác nhau giữa nhánh hiện tại và commit gần nhất.

## **10. Git log**

```bash
git log
git log --oneline
git log --oneline --graph
```

- `git log` dùng để liệt kê tất cả các commit trong repo hiện tại.
- `git log --oneline` dùng để liệt kê tất cả các commit trong repo hiện tại trên 1 dòng.
- `git log --oneline --graph` dùng để liệt kê tất cả các commit trong repo hiện tại trên 1 dòng và vẽ đồ thị.

## **11. Làm việc với nhánh**

### **Git Branch**

```bash
git branch
git branch branch_name
git branch -m branch_name new_branch_name
git branch -d branch_name
git branch -D branch_name
```

- Khi đang làm việc với một nhánh nào đó, nếu muốn tạo một nhánh mới thì cần checkout về nhánh mình muốn base trước. Vì git branch sẽ tạo nhánh dựa trên nhánh hiện tại.
- `git branch` dùng để liệt kê toàn bộ các branch trong repo hiện tại.
- `git branch branch_name` dùng để tạo một branch mới trong repo hiện tại.
- `git branch -m branch_name new_branch_name` dùng để đổi tên một branch trong repo hiện tại.
- `git branch -d branch_name` dùng để xóa một branch trong repo hiện tại.
- `git branch -D branch_name` dùng để xóa một branch trong repo hiện tại mà không cần kiểm tra xem branch đó đã được merge vào branch khác hay chưa.

### **Git Checkout**

```bash
git checkout branch_name
git checkout -b branch_name
```

- `git checkout branch_name` dùng để chuyển sang một branch khác trong repo hiện tại.
- `git checkout -b branch_name` dùng để tạo một branch mới và chuyển sang branch đó.

### **Git Merge**

```bash
git merge branch_name
```

- `git merge branch_name` dùng để merge branch_name vào branch hiện tại.

### **Git Rebase**

```bash
git rebase branch_name
```

- `git rebase branch_name` dùng để rebase branch hiện tại vào branch_name.
- Cách thức hoạt động của `git rebase` là tìm commit chung gần nhất giữa branch hiện tại và branch_name, sau đó tạo một commit mới chứa các thay đổi của branch hiện tại và đặt commit mới này lên commit chung gần nhất.
- Sau khi done 1 branch `feat`, có thể rebase dev vào `feat`, sau đó checkout về `dev` và merge `feat` vào `dev`.
- Với rebase, khi ở nhánh master có thêm commit mới, ta sẽ dùng rebase để thực hiện merge các commits mới này vào nhánh feature của mình.
- Với merge, ta sẽ dùng khi muốn merge những commits từ feature vào master. Rất đơn giản và gọn gàng
- **Dùng `git rebase` sẽ luôn đi cùng `git push -f`** vì khi rebase, local repository có history khác với remote repository. Git không tự động ghi đè commit trên remote bằng commit mới có history khác của bạn.
- Vì thế, bạn cần dùng git push -f (force push) để buộc git ghi đè commit cũ trên remote bằng commit mới sau khi rebase.
- Nên hãy luôn nhớ pull cả 2 nhánh master và feature về trước khi thực hiện rebase để tránh miss code

### **Git Cherry-pick**

```bash
git cherry-pick commit_hash
```

- `git cherry-pick commit_hash` cho phép áp dụng commit từ 1 nhánh này sang 1 nhánh khác
  - Chọn 1 commit bất kỳ từ 1 nhánh
  - Sử dụng git cherry-pick để áp dụng chính xác các thay đổi đó vào nhánh khác hiện tại
  - Kết quả là 1 commit mới (với cùng nội dung) được tạo ra ở nhánh hiện tại

- Ứng dụng:
  - Để áp dụng các thay đổi từ 1 nhánh sang nhánh khác
  - Fix 1 bug từ 1 nhánh và áp dụng lên nhiều nhánh khác.

### **Git Stash**

```bash
git stash
git stash list
git stash apply
git stash drop
git stash pop
```

- Git stash là một tính năng cho phép bạn lưu trữ (stash away) các thay đổi chưa được commit trong working directory rồi sau đó mang trở lại khi cần.
- `git stash` dùng để lưu trữ các thay đổi chưa được commit trong working directory.
- `git stash list` dùng để liệt kê tất cả các stash trong repo hiện tại.
- `git stash apply` dùng để áp dụng và giữ lại stash cuối cùng trong repo hiện tại.
- `git stash drop` dùng để xóa stash cuối cùng trong repo hiện tại.
- `git stash pop` dùng để áp dụng và xóa stash cuối cùng trong repo hiện tại.
- **Lưu ý:** git stash hoạt động trên toàn bộ repo, không phải chỉ trên một branch. Do đó, nếu bạn đang ở branch `feat` và thực hiện `git stash`, thì khi checkout sang branch `dev` và thực hiện `git stash apply` thì các thay đổi của branch `feat` sẽ được áp dụng vào branch `dev`.

## **12. Remote**

### **Git Fetch**

- `git fetch <remote>`: Tìm nạp tất cả các branch từ kho lưu trữ. Điều này cũng tải xuống tất cả các commit và tệp được yêu cầu từ kho lưu trữ khác.

- `git fetch <remote> <branch>`: Tương tự như lệnh trên, nhưng chỉ tìm nạp những branch được chỉ định.

- `git fetch --all`: Tìm nạp tất cả các remote đã đăng ký và các branch của chúng.

### **Git Pull**

```bash
git pull
git pull origin branch_name
```

- `git pull` dùng để pull các thay đổi từ remote repository vào local repository.
- `git pull origin branch_name` dùng để pull các thay đổi từ remote repo vào local repo, chỉ pull nhánh branch_name.
- `git pull` là một câu lệnh kết hợp của `git fetch` và `git merge`. Nó sẽ tự động fetch các thay đổi từ remote repository vào local repository và merge chúng.

### **Git Push**

```bash
git push
git push origin master
```

- `git push` dùng để push các thay đổi từ local repository vào remote repository.
- `git push origin master` dùng để push các thay đổi từ local repository vào remote repository, chỉ push nhánh master.

### **Git Tag**

```bash
git tag -a v1.0 -m "Version 1.0"
git tag -d v1.0
git push origin v1.0
```

- `git tag -a v1.0 -m "Version 1.0"` dùng để tạo một tag mới trong repo hiện tại.
- `git tag -d v1.0` dùng để xóa một tag trong repo hiện tại
- `git push origin v1.0` dùng để push một tag từ local repo lên remote repo.

### **Ignoring Patterns**

```bash
git check-ignore -v file_name
```

- Git ignore là một tính năng cho phép bạn bỏ qua (ignore) các file không cần theo dõi trong repo hiện tại.
- `git check-ignore -v file_name` dùng để kiểm tra xem file_name có bị ignore hay không.

### **Tạo pull/merge request**

- Cách tạo pull request:
  - Checkout sang branch cần tạo pull request: `git checkout feature/abc`
  - Push code lên branch: `git push origin feature/abc`
  - Tạo pull request từ branch feature/abc đến develop băng cách vào tab pull request trên Github/Gitlab

- Why?
  - Để có thể review code trước khi merge vào develop/master/
  - Do đó **không được** merge trực tiếp code vào develop/master rồi push lên mà phải tạo pull request

- Cách review pull request:
  - Vào tab pull request trên Github/Gitlab
  - Chọn pull request cần review
  - Kiểm tra code, comment, approve hoặc reject pull request

### **Tạo release**

- Cách tạo release như sau:
  - Checkout sang branch develop: `git checkout develop`
  - Rebase develop về master: `git rebase master`
  - Checkout lại branch master: `git checkout master`
  - Merge develop vào master: `git merge develop`
  - Push code lên master: `git push origin master`
  - Tạo release trên Github/Gitlab

### **Issue**

- Issue trong Git là cách để báo cáo vấn đề, lỗi hoặc yêu cầu tính năng mới với dự án.

- Có thể sử dụng Issue để phân chia công việc thành các task và mỗi thành viên đảm nhiệm một Issue.
- Convention chung của team:
  - 1. Tạo 1 task trên Notion: brief describe nội dung cần làm
  - 2. Tạo issue trên gitlab - Lưu lại link của Issue vào Notion ở bước 1
  - 3. Tạo branch feat/1-tên-issue (giả sử index của issue là 1)
  - 4. Hoàn thành công việc và commit đầy đủ. Trong message commit phải có link của issue ở bước 2, ví dụ "#1 - Fix bug"
  - 5. Tạo MR từ nhánh feat/1-tên-issue về main (or master) - Lưu link MR vào Notion ở bước 1
  - 6. Báo cho mentor review MR

## **13. Git WorkFlow**

- Mô tả tổng quan về Gitflow Workflow.

![gitwf](https://images.viblo.asia/84f47fd1-a009-4beb-8957-26395fe1023d.png)

- Master branch: Chứa code ổn định nhất, có các version đã sẵn sàng để realease cho người dùng. Tất cả các thay đổi trên code sẽ được merge vào nhánh này sau khi đã được test và review.

![master](https://images.viblo.asia/f71e46bd-452f-48c1-b451-2f8a25fff458.png)

- Develop: là merge code của tất cả các branchs feature. Khi dev team hoàn thành hết feature của một topic, teamlead sẽ review ứng dụng và merge đến branchs release để tạo ra bản một bản release cho sản phẩm.

![dev](https://images.viblo.asia/6e91e85b-3152-4a04-a04d-160aa0bd5135.png)

- Feature: Được base trên branchs Develop. Mỗi khi phát triển một feature mới chúng ta cần tạo một branchs mới base trên branchs Develop để code cho feature đó. Sau khi code xong, tạo merge request đến branchs develop để teamlead review mà merge lại vào branchs Develop.

```bash
git checkout develop
git checkout -b feature_branch
# after done and reviewed
git checkout develop
git merge feature_branch
```

![feat](https://images.viblo.asia/e4f9e958-2d5e-4f9e-98bc-9c77453b5983.png)

- Release: Tạo ra từ develop branch. Dùng để chuẩn bị cho một release (tạo ra bản build để test và kiểm tra). Sau khi test sẽ merge vào cả master và develop, sau đó xóa branchs release.

```bash
git checkout develop
git checkout -b release/0.1.0
# after done test and ready to release
git checkout main
git merge release/0.1.0
git checkout develop
git merge release/0.1.0
git branch -d release/0.1.0
```

![rls](https://images.viblo.asia/7b05bf3e-e652-4ef5-817d-bef89314ef7c.png)

- Hotfix: Được base trên nhánh master để sửa nhanh những lỗi trên productions. Sau khi fix sẽ merge vào cả master và develop. (vì không được phép sửa trực tiếp trên branch master nên phải làm cách này)

```bash
git checkout main
git checkout -b hotfix_branch
# after done fix
git checkout main
git merge hotfix_branch
git checkout develop
git merge hotfix_branch
git branch -D hotfix_branch
```

![hotfix](https://images.viblo.asia/9fab3a45-1282-4b45-9db9-f80dc7143ae5.png)

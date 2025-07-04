# 文本浏览器模块 - 基于Microsoft Autogen团队的实现
# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py

# 导入必要的标准库模块
import mimetypes  # 用于MIME类型检测
import os  # 操作系统接口
import pathlib  # 面向对象的文件系统路径
import re  # 正则表达式操作
import time  # 时间相关功能
import uuid  # UUID生成
from typing import Any, Dict, List, Optional, Tuple, Union  # 类型提示
from urllib.parse import unquote, urljoin, urlparse  # URL解析工具

# 导入第三方库
import pathvalidate  # 路径验证工具
import requests  # HTTP请求库

# 导入内部模块
from smolagents import Tool  # 智能代理工具基类

# 导入本地模块
from .cookies import COOKIES  # Cookie配置
from .mdconvert import FileConversionException, MarkdownConverter, UnsupportedFormatException  # Markdown转换器


class SimpleTextBrowser:
    """
    简单的文本浏览器类 - 类似于Lynx的极简文本浏览器，适用于AI代理使用

    这个类提供了基本的网页浏览功能，包括：
    - 访问网页并转换为文本格式
    - 分页显示内容
    - 页面内搜索
    - 文件下载
    - 搜索引擎集成
    """

    def __init__(
        self,
        start_page: Optional[str] = None,  # 起始页面URL
        viewport_size: Optional[int] = 1024 * 8,  # 视口大小（字符数）
        downloads_folder: Optional[Union[str, None]] = None,  # 下载文件夹路径
        serpapi_key: Optional[Union[str, None]] = None,  # SerpAPI密钥
        serper_api_key: Optional[Union[str, None]] = None,  # Serper API密钥
        request_kwargs: Optional[Union[Dict[str, Any], None]] = None,  # HTTP请求参数
    ):
        """
        初始化简单文本浏览器

        Args:
            start_page: 起始页面URL，默认为"about:blank"
            viewport_size: 视口大小，控制每页显示的字符数
            downloads_folder: 下载文件保存的文件夹路径
            serpapi_key: SerpAPI的API密钥，用于Google搜索
            serper_api_key: Serper API的密钥，用于Google搜索
            request_kwargs: HTTP请求的额外参数
        """
        # 设置起始页面，如果未指定则使用空白页
        self.start_page: str = start_page if start_page else "about:blank"
        # 设置视口大小，仅适用于标准URI类型
        self.viewport_size = viewport_size
        # 设置下载文件夹路径
        self.downloads_folder = downloads_folder
        # 初始化浏览历史记录，存储(URL, 时间戳)元组
        self.history: List[Tuple[str, float]] = list()
        # 当前页面标题
        self.page_title: Optional[str] = None
        # 当前视口页面索引
        self.viewport_current_page = 0
        # 视口页面边界列表，存储(开始位置, 结束位置)元组
        self.viewport_pages: List[Tuple[int, int]] = list()
        # 设置初始地址
        self.set_address(self.start_page)
        # 设置搜索API密钥
        self.serpapi_key = serpapi_key
        self.serper_api_key = serper_api_key
        # 设置HTTP请求参数
        self.request_kwargs = request_kwargs
        self.request_kwargs["cookies"] = COOKIES
        # 初始化Markdown转换器
        self._mdconvert = MarkdownConverter()
        # 当前页面内容
        self._page_content: str = ""

        # 页面内搜索相关变量
        self._find_on_page_query: Union[str, None] = None  # 当前搜索查询
        self._find_on_page_last_result: Union[int, None] = None  # 最后一次搜索结果的位置

    @property
    def address(self) -> str:
        """
        获取当前页面的地址

        Returns:
            str: 当前页面的URL地址
        """
        return self.history[-1][0]

    def set_address(self, uri_or_path: str, filter_year: Optional[int] = None) -> None:
        """
        设置浏览器地址并加载页面

        Args:
            uri_or_path: 要访问的URI或路径
            filter_year: 可选的年份过滤器，用于搜索结果过滤
        """
        # TODO: 处理锚点链接
        # 将新地址添加到历史记录中，记录访问时间
        self.history.append((uri_or_path, time.time()))

        # 处理特殊URI
        if uri_or_path == "about:blank":
            # 空白页面
            self._set_page_content("")
        elif uri_or_path.startswith("google:"):
            # Google搜索请求
            if self.serpapi_key:
                # 使用SerpAPI进行搜索
                self._serpapi_search(uri_or_path[len("google:") :].strip(), filter_year=filter_year)
            elif self.serper_api_key:
                # 使用Serper API进行搜索
                self._serper_search(uri_or_path[len("google:") :].strip())
        else:
            # 处理普通URL
            if (
                not uri_or_path.startswith("http:")
                and not uri_or_path.startswith("https:")
                and not uri_or_path.startswith("file:")
            ):
                # 如果是相对路径，则基于前一个地址构建完整URL
                if len(self.history) > 1:
                    prior_address = self.history[-2][0]
                    uri_or_path = urljoin(prior_address, uri_or_path)
                    # 更新历史记录中的地址为完整路径
                    self.history[-1] = (uri_or_path, self.history[-1][1])
            # 获取页面内容
            self._fetch_page(uri_or_path)

        # 重置视口状态
        self.viewport_current_page = 0
        self.find_on_page_query = None
        self.find_on_page_viewport = None

    @property
    def viewport(self) -> str:
        """
        获取当前视口的内容

        Returns:
            str: 当前视口显示的页面内容
        """
        bounds = self.viewport_pages[self.viewport_current_page]
        return self.page_content[bounds[0] : bounds[1]]

    @property
    def page_content(self) -> str:
        """
        获取当前页面的完整内容

        Returns:
            str: 当前页面的完整文本内容
        """
        return self._page_content

    def _set_page_content(self, content: str) -> None:
        """
        设置当前页面的文本内容

        Args:
            content: 要设置的页面内容
        """
        self._page_content = content
        # 重新分页
        self._split_pages()
        # 确保当前视口页面索引有效
        if self.viewport_current_page >= len(self.viewport_pages):
            self.viewport_current_page = len(self.viewport_pages) - 1

    def page_down(self) -> None:
        """向下翻页，移动到下一个视口页面"""
        self.viewport_current_page = min(self.viewport_current_page + 1, len(self.viewport_pages) - 1)

    def page_up(self) -> None:
        """向上翻页，移动到上一个视口页面"""
        self.viewport_current_page = max(self.viewport_current_page - 1, 0)

    def find_on_page(self, query: str) -> Union[str, None]:
        """
        在页面中搜索指定查询字符串，从当前视口开始向前搜索，必要时循环回到开始位置

        Args:
            query: 要搜索的查询字符串

        Returns:
            Union[str, None]: 如果找到匹配项则返回包含匹配项的视口内容，否则返回None
        """
        # 检查是否是使用相同查询的连续搜索
        # 如果是，则映射到find_next方法
        if query == self._find_on_page_query and self.viewport_current_page == self._find_on_page_last_result:
            return self.find_next()

        # 这是一个新的搜索，从当前视口开始
        self._find_on_page_query = query
        viewport_match = self._find_next_viewport(query, self.viewport_current_page)
        if viewport_match is None:
            # 未找到匹配项
            self._find_on_page_last_result = None
            return None
        else:
            # 找到匹配项，更新当前视口
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def find_next(self) -> Union[str, None]:
        """
        滚动到下一个匹配查询的视口

        Returns:
            Union[str, None]: 如果找到下一个匹配项则返回视口内容，否则返回None
        """
        if self._find_on_page_query is None:
            return None

        # 确定搜索起始位置
        starting_viewport = self._find_on_page_last_result
        if starting_viewport is None:
            starting_viewport = 0
        else:
            starting_viewport += 1
            # 如果超出范围，则从头开始
            if starting_viewport >= len(self.viewport_pages):
                starting_viewport = 0

        # 搜索下一个匹配的视口
        viewport_match = self._find_next_viewport(self._find_on_page_query, starting_viewport)
        if viewport_match is None:
            # 未找到更多匹配项
            self._find_on_page_last_result = None
            return None
        else:
            # 找到匹配项，更新当前视口
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def _find_next_viewport(self, query: str, starting_viewport: int) -> Union[int, None]:
        """
        从指定的起始视口开始搜索匹配项，到达末尾时循环搜索

        Args:
            query: 搜索查询字符串
            starting_viewport: 起始视口索引

        Returns:
            Union[int, None]: 找到匹配项的视口索引，未找到则返回None
        """
        if query is None:
            return None

        # 规范化查询字符串，并转换为正则表达式
        nquery = re.sub(r"\*", "__STAR__", query)  # 临时替换通配符
        nquery = " " + (" ".join(re.split(r"\W+", nquery))).strip() + " "  # 分词并重组
        nquery = nquery.replace(" __STAR__ ", "__STAR__ ")  # 将孤立的星号与前面的词合并
        nquery = nquery.replace("__STAR__", ".*").lower()  # 将通配符转换为正则表达式

        if nquery.strip() == "":
            return None

        # 构建搜索索引列表，从起始视口到末尾，然后从开头到起始视口
        idxs = list()
        idxs.extend(range(starting_viewport, len(self.viewport_pages)))
        idxs.extend(range(0, starting_viewport))

        # 在每个视口中搜索匹配项
        for i in idxs:
            bounds = self.viewport_pages[i]
            content = self.page_content[bounds[0] : bounds[1]]

            # TODO: 移除markdown链接和图片
            # 规范化内容进行搜索
            ncontent = " " + (" ".join(re.split(r"\W+", content))).strip().lower() + " "
            if re.search(nquery, ncontent):
                return i

        return None

    def visit_page(self, path_or_uri: str, filter_year: Optional[int] = None) -> str:
        """
        访问指定页面并返回视口内容

        Args:
            path_or_uri: 要访问的路径或URI
            filter_year: 可选的年份过滤器

        Returns:
            str: 当前视口的内容
        """
        self.set_address(path_or_uri, filter_year=filter_year)
        return self.viewport

    def _split_pages(self) -> None:
        """
        将页面内容分割成多个视口页面
        """
        # 不分割搜索结果页面
        if self.address.startswith("google:"):
            self.viewport_pages = [(0, len(self._page_content))]
            return

        # 处理空页面
        if len(self._page_content) == 0:
            self.viewport_pages = [(0, 0)]
            return

        # 将视口分割成多个页面
        self.viewport_pages = []
        start_idx = 0
        while start_idx < len(self._page_content):
            # 计算页面结束位置
            end_idx = min(start_idx + self.viewport_size, len(self._page_content))  # type: ignore[operator]
            # 调整到在空格处结束，避免截断单词
            while end_idx < len(self._page_content) and self._page_content[end_idx - 1] not in [" ", "\t", "\r", "\n"]:
                end_idx += 1
            # 添加页面边界
            self.viewport_pages.append((start_idx, end_idx))
            start_idx = end_idx

    def _serpapi_search(self, query: str, filter_year: Optional[int] = None) -> None:
        """
        使用SerpAPI执行Google搜索

        Args:
            query: 搜索查询字符串
            filter_year: 可选的年份过滤器

        Raises:
            ValueError: 当SerpAPI密钥缺失时
            Exception: 当搜索无结果时
        """
        if self.serpapi_key is None:
            raise ValueError("Missing SerpAPI key.")

        # 构建搜索参数
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
        }
        # 如果指定了年份过滤器，添加时间范围参数
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        # 执行搜索
        search = GoogleSearch(params)
        results = search.get_dict()
        self.page_title = f"{query} - Search"

        # 检查搜索结果
        if "organic_results" not in results.keys():
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        if len(results["organic_results"]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            self._set_page_content(
                f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."
            )
            return

        def _prev_visit(url):
            """检查URL是否之前访问过，并返回访问时间信息"""
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i][0] == url:
                    return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
            return ""

        # 处理搜索结果
        web_snippets: List[str] = list()
        idx = 0
        if "organic_results" in results:
            for page in results["organic_results"]:
                idx += 1
                # 提取发布日期
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                # 提取来源信息
                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                # 提取摘要信息
                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                # 构建搜索结果条目
                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{_prev_visit(page['link'])}{snippet}"

                # 清理无用信息
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

        # 构建最终的搜索结果页面内容
        content = (
            f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )

        self._set_page_content(content)

    def _fetch_page(self, url: str) -> None:
        """
        获取指定URL的页面内容

        Args:
            url: 要获取的页面URL
        """
        download_path = ""
        try:
            if url.startswith("file://"):
                # 处理本地文件URL
                download_path = os.path.normcase(os.path.normpath(unquote(url[7:])))
                res = self._mdconvert.convert_local(download_path)
                self.page_title = res.title
                self._set_page_content(res.text_content)
            else:
                # 处理HTTP/HTTPS URL
                # 准备请求参数
                request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}
                request_kwargs["stream"] = True

                # 发送HTTP请求
                response = requests.get(url, **request_kwargs)
                response.raise_for_status()

                # 如果HTTP请求成功
                content_type = response.headers.get("content-type", "")

                # 处理文本或HTML内容
                if "text/" in content_type.lower():
                    res = self._mdconvert.convert_response(response)
                    self.page_title = res.title
                    self._set_page_content(res.text_content)
                # 处理下载文件
                else:
                    # 尝试生成安全的文件名
                    fname = None
                    download_path = None
                    try:
                        fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                        # 如果文件已存在，添加后缀
                        suffix = 0
                        while os.path.exists(download_path) and suffix < 1000:
                            suffix += 1
                            base, ext = os.path.splitext(fname)
                            new_fname = f"{base}__{suffix}{ext}"
                            download_path = os.path.abspath(os.path.join(self.downloads_folder, new_fname))

                    except NameError:
                        pass

                    # 如果无法生成合适的文件名，则创建一个
                    if fname is None:
                        extension = mimetypes.guess_extension(content_type)
                        if extension is None:
                            extension = ".download"
                        fname = str(uuid.uuid4()) + extension
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                    # 打开文件进行写入
                    with open(download_path, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=512):
                            fh.write(chunk)

                    # 渲染下载的文件
                    local_uri = pathlib.Path(download_path).as_uri()
                    self.set_address(local_uri)

        except UnsupportedFormatException as e:
            # 处理不支持的文件格式异常
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileConversionException as e:
            # 处理文件转换异常
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileNotFoundError:
            # 处理文件未找到异常
            self.page_title = "Error 404"
            self._set_page_content(f"## Error 404\n\nFile not found: {download_path}")
        except requests.exceptions.RequestException as request_exception:
            # 处理HTTP请求异常
            try:
                self.page_title = f"Error {response.status_code}"

                # 如果错误以HTML格式呈现，我们也可以渲染它
                content_type = response.headers.get("content-type", "")
                if content_type is not None and "text/html" in content_type.lower():
                    res = self._mdconvert.convert(response)
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{res.text_content}")
                else:
                    # 处理非HTML错误响应
                    text = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        text += chunk
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{text}")
            except NameError:
                # 处理response变量未定义的情况
                self.page_title = "Error"
                self._set_page_content(f"## Error\n\n{str(request_exception)}")

    def _state(self) -> Tuple[str, str]:
        """
        获取浏览器当前状态信息

        Returns:
            Tuple[str, str]: 包含页面头部信息和当前视口内容的元组
        """
        # 构建页面头部信息
        header = f"Address: {self.address}\n"
        if self.page_title is not None:
            header += f"Title: {self.page_title}\n"

        current_page = self.viewport_current_page
        total_pages = len(self.viewport_pages)

        # 检查是否之前访问过此页面
        address = self.address
        for i in range(len(self.history) - 2, -1, -1):  # 从倒数第二个开始
            if self.history[i][0] == address:
                header += f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
                break

        # 添加视口位置信息
        header += f"Viewport position: Showing page {current_page + 1} of {total_pages}.\n"
        return (header, self.viewport)


class SearchInformationTool(Tool):
    """
    网络搜索工具类 - 执行网络搜索查询并返回搜索结果
    """
    name = "web_search"
    description = "Perform a web search query (think a google search) and returns the search results."
    inputs = {"query": {"type": "string", "description": "The web search query to perform."}}
    inputs["filter_year"] = {
        "type": "string",
        "description": "[Optional parameter]: filter the search results to only include pages from a specific year. For example, '2020' will only include pages from 2020. Make sure to use this parameter if you're trying to search for articles from a specific date!",
        "nullable": True,
    }
    output_type = "string"

    def __init__(self, browser):
        """
        初始化搜索工具

        Args:
            browser: SimpleTextBrowser实例
        """
        super().__init__()
        self.browser = browser

    def forward(self, query: str, filter_year: Optional[int] = None) -> str:
        """
        执行网络搜索

        Args:
            query: 搜索查询字符串
            filter_year: 可选的年份过滤器

        Returns:
            str: 格式化的搜索结果
        """
        self.browser.visit_page(f"google: {query}", filter_year=filter_year)
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class VisitTool(Tool):
    """
    页面访问工具类 - 访问指定URL的网页并返回其文本内容
    """
    name = "visit_page"
    description = "Visit a webpage at a given URL and return its text. Given a url to a YouTube video, this returns the transcript."
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the webapge to visit."}}
    output_type = "string"

    def __init__(self, browser):
        """
        初始化页面访问工具

        Args:
            browser: SimpleTextBrowser实例
        """
        super().__init__()
        self.browser = browser

    def forward(self, url: str) -> str:
        """
        访问指定URL的页面

        Args:
            url: 要访问的页面URL

        Returns:
            str: 页面的文本内容
        """
        self.browser.visit_page(url)
        # header, content = self.browser._state()
        # return header.strip() + "\n=======================\n" + content
        return self.browser.page_content


class DownloadTool(Tool):
    """
    文件下载工具类 - 下载指定URL的文件
    """
    name = "download_file"
    description = """
Download a file at a given URL. The file should be of this format: [".xlsx", ".pptx", ".wav", ".mp3", ".png", ".docx"]
After using this tool, for further inspection of this page you should return the download path to your manager via final_answer, and they will be able to inspect it.
DO NOT use this tool for .pdf or .txt or .htm files: for these types of files use visit_page with the file url instead."""
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the file to be downloaded."}}
    output_type = "string"

    def __init__(self, browser):
        """
        初始化文件下载工具

        Args:
            browser: SimpleTextBrowser实例
        """
        super().__init__()
        self.browser = browser

    def forward(self, url: str) -> str:
        """
        下载指定URL的文件

        Args:
            url: 要下载的文件URL

        Returns:
            str: 下载完成的消息，包含文件保存路径

        Raises:
            Exception: 当尝试下载PDF、TXT或HTML文件时
        """
        # 处理arXiv URL，将abstract链接转换为PDF链接
        if "arxiv" in url:
            url = url.replace("abs", "pdf")

        # 发送HTTP请求获取文件
        response = requests.get(url)
        content_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type)

        # 确定文件保存路径
        if extension and isinstance(extension, str):
            new_path = f"./downloads/file{extension}"
        else:
            new_path = "./downloads/file.object"

        # 保存文件到本地
        with open(new_path, "wb") as f:
            f.write(response.content)

        # 检查文件类型，确保不是应该使用visit_page的文件类型
        if "pdf" in extension or "txt" in extension or "htm" in extension:
            raise Exception("Do not use this tool for pdf or txt or html files: use visit_page instead.")

        return f"File was downloaded and saved under path {new_path}."


class ArchiveSearchTool(Tool):
    """
    网页存档搜索工具类 - 在Wayback Machine中搜索指定URL的存档版本
    """
    name = "find_archived_url"
    description = "Given a url, searches the Wayback Machine and returns the archived version of the url that's closest in time to the desired date."
    inputs = {
        "url": {"type": "string", "description": "The url you need the archive for."},
        "date": {
            "type": "string",
            "description": "The date that you want to find the archive for. Give this date in the format 'YYYYMMDD', for instance '27 June 2008' is written as '20080627'.",
        },
    }
    output_type = "string"

    def __init__(self, browser):
        """
        初始化存档搜索工具

        Args:
            browser: SimpleTextBrowser实例
        """
        super().__init__()
        self.browser = browser

    def forward(self, url, date) -> str:
        """
        搜索指定URL在指定日期的存档版本

        Args:
            url: 要搜索存档的URL
            date: 目标日期，格式为YYYYMMDD

        Returns:
            str: 存档页面的内容

        Raises:
            Exception: 当URL未在Wayback Machine中存档时
        """
        # 构建Wayback Machine API请求URL
        no_timestamp_url = f"https://archive.org/wayback/available?url={url}"
        archive_url = no_timestamp_url + f"&timestamp={date}"

        # 发送API请求
        response = requests.get(archive_url).json()
        response_notimestamp = requests.get(no_timestamp_url).json()

        # 查找最接近的存档快照
        if "archived_snapshots" in response and "closest" in response["archived_snapshots"]:
            closest = response["archived_snapshots"]["closest"]
            print("Archive found!", closest)
        elif "archived_snapshots" in response_notimestamp and "closest" in response_notimestamp["archived_snapshots"]:
            closest = response_notimestamp["archived_snapshots"]["closest"]
            print("Archive found!", closest)
        else:
            raise Exception(f"Your {url=} was not archived on Wayback Machine, try a different url.")

        # 访问存档页面
        target_url = closest["url"]
        self.browser.visit_page(target_url)
        header, content = self.browser._state()
        return (
            f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n"
            + header.strip()
            + "\n=======================\n"
            + content
        )


class PageUpTool(Tool):
    """
    页面向上滚动工具类 - 将视口向上滚动一页
    """
    name = "page_up"
    description = "Scroll the viewport UP one page-length in the current webpage and return the new viewport content."
    inputs = {}
    output_type = "string"

    def __init__(self, browser):
        """
        初始化页面向上滚动工具

        Args:
            browser: SimpleTextBrowser实例
        """
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        """
        执行向上滚动操作

        Returns:
            str: 滚动后的视口内容
        """
        self.browser.page_up()
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class PageDownTool(Tool):
    """
    页面向下滚动工具类 - 将视口向下滚动一页
    """
    name = "page_down"
    description = (
        "Scroll the viewport DOWN one page-length in the current webpage and return the new viewport content."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, browser):
        """
        初始化页面向下滚动工具

        Args:
            browser: SimpleTextBrowser实例
        """
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        """
        执行向下滚动操作

        Returns:
            str: 滚动后的视口内容
        """
        self.browser.page_down()
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class FinderTool(Tool):
    """
    页面内搜索工具类 - 在当前页面中搜索指定字符串（类似Ctrl+F功能）
    """
    name = "find_on_page_ctrl_f"
    description = "Scroll the viewport to the first occurrence of the search string. This is equivalent to Ctrl+F."
    inputs = {
        "search_string": {
            "type": "string",
            "description": "The string to search for on the page. This search string supports wildcards like '*'",
        }
    }
    output_type = "string"

    def __init__(self, browser):
        """
        初始化页面内搜索工具

        Args:
            browser: SimpleTextBrowser实例
        """
        super().__init__()
        self.browser = browser

    def forward(self, search_string: str) -> str:
        """
        在页面中搜索指定字符串

        Args:
            search_string: 要搜索的字符串，支持通配符'*'

        Returns:
            str: 包含搜索结果的视口内容，如果未找到则返回错误消息
        """
        find_result = self.browser.find_on_page(search_string)
        header, content = self.browser._state()

        if find_result is None:
            return (
                header.strip()
                + f"\n=======================\nThe search string '{search_string}' was not found on this page."
            )
        else:
            return header.strip() + "\n=======================\n" + content


class FindNextTool(Tool):
    """
    查找下一个匹配项工具类 - 滚动到下一个匹配搜索字符串的位置
    """
    name = "find_next"
    description = "Scroll the viewport to next occurrence of the search string. This is equivalent to finding the next match in a Ctrl+F search."
    inputs = {}
    output_type = "string"

    def __init__(self, browser):
        """
        初始化查找下一个匹配项工具

        Args:
            browser: SimpleTextBrowser实例
        """
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        """
        查找并滚动到下一个匹配项

        Returns:
            str: 包含下一个匹配项的视口内容，如果未找到则返回错误消息
        """
        find_result = self.browser.find_next()
        header, content = self.browser._state()

        if find_result is None:
            return header.strip() + "\n=======================\nThe search string was not found on this page."
        else:
            return header.strip() + "\n=======================\n" + content